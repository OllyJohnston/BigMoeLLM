#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bmoe {

// Adaptive Replacement Cache (ARC) with optional pinned item support.
//
// Self-tuning cache balancing recency (T1) and frequency (T2) using ghost histories (B1, B2)
// based on Megiddo & Modha (FAST '03).
//
// Pinned items (e.g., top 20% grammatical/syntactic router weights) are never evicted
// during cache replacement sweeps.
template <typename Key, typename Value = void>
class AdaptiveReplacementCache {
public:
    enum class Location {
        None,
        T1, // Top recency (resident)
        T2, // Top frequency (resident)
        B1, // Bottom recency ghost (metadata only)
        B2  // Bottom frequency ghost (metadata only)
    };

    struct LookupResult {
        bool hit = false;
        Location prev_location = Location::None;
        std::vector<Key> evicted_resident_keys; // Resident keys whose physical data was evicted
    };

    explicit AdaptiveReplacementCache(size_t capacity = 0)
        : capacity_(capacity), target_p_(0.0) {}

    void set_capacity(size_t capacity) {
        std::lock_guard<std::mutex> lk(mtx_);
        capacity_ = capacity;
        if (target_p_ > (double) capacity_) {
            target_p_ = (double) capacity_;
        }
    }

    size_t capacity() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return capacity_;
    }

    double target_p() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return target_p_;
    }

    size_t resident_size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return t1_.size() + t2_.size();
    }

    size_t ghost_size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return b1_.size() + b2_.size();
    }

    size_t pinned_size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return pinned_.size();
    }

    bool is_resident(const Key & key) const {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = entries_.find(key);
        if (it == entries_.end()) return false;
        return it->second.location == Location::T1 || it->second.location == Location::T2;
    }

    bool is_pinned(const Key & key) const {
        std::lock_guard<std::mutex> lk(mtx_);
        return pinned_.find(key) != pinned_.end();
    }

    void pin(const Key & key) {
        std::lock_guard<std::mutex> lk(mtx_);
        pinned_.insert(key);
    }

    void unpin(const Key & key) {
        std::lock_guard<std::mutex> lk(mtx_);
        pinned_.erase(key);
    }

    // Records a reference/access to `key`.
    // Returns whether it was a resident hit, its previous location, and any keys whose
    // physical data was evicted to make room.
    LookupResult access(const Key & key) {
        std::lock_guard<std::mutex> lk(mtx_);
        LookupResult res;
        if (capacity_ == 0) return res;

        auto it = entries_.find(key);
        if (it == entries_.end()) {
            // Case 4: Miss in all (T1 U T2 U B1 U B2)
            res.hit = false;
            res.prev_location = Location::None;


            const size_t l1_sz = t1_.size() + b1_.size();
            const size_t total_sz = t1_.size() + t2_.size() + b1_.size() + b2_.size();

            if (l1_sz == capacity_) {
                if (t1_.size() < capacity_) {
                    // Evict oldest in B1
                    evict_ghost_b1();
                    replace(key, res.evicted_resident_keys);
                } else {
                    // Evict oldest in T1
                    evict_lru_t1(res.evicted_resident_keys, false /* don't move to B1 */);
                }
            } else if (l1_sz < capacity_) {
                if (total_sz >= capacity_) {
                    if (total_sz == 2 * capacity_) {
                        evict_ghost_b2();
                    }
                    replace(key, res.evicted_resident_keys);
                }
            }

            // Insert into MRU of T1
            t1_.push_front(key);
            Entry e;
            e.location = Location::T1;
            e.it = t1_.begin();
            entries_[key] = e;
            return res;
        }

        Entry & entry = it->second;
        res.prev_location = entry.location;

        if (entry.location == Location::T1 || entry.location == Location::T2) {
            // Case 1: Resident Cache Hit (in T1 or T2) -> Move to MRU of T2
            res.hit = true;
            remove_from_list(entry.location, entry.it);
            t2_.push_front(key);
            entry.location = Location::T2;
            entry.it = t2_.begin();
            return res;
        }

        if (entry.location == Location::B1) {
            // Case 2: Hit in B1 ghost cache
            res.hit = false;
            const double delta = b1_.size() >= b2_.size() ? 1.0 : ((double) b2_.size() / (double) b1_.size());
            target_p_ = std::min((double) capacity_, target_p_ + delta);

            replace(key, res.evicted_resident_keys);

            remove_from_list(Location::B1, entry.it);
            t2_.push_front(key);
            entry.location = Location::T2;
            entry.it = t2_.begin();
            return res;
        }

        if (entry.location == Location::B2) {
            // Case 3: Hit in B2 ghost cache
            res.hit = false;
            const double delta = b2_.size() >= b1_.size() ? 1.0 : ((double) b1_.size() / (double) b2_.size());
            target_p_ = std::max(0.0, target_p_ - delta);

            replace(key, res.evicted_resident_keys);

            remove_from_list(Location::B2, entry.it);
            t2_.push_front(key);
            entry.location = Location::T2;
            entry.it = t2_.begin();
            return res;
        }

        return res;
    }

    // Pin top N percentage of most frequently accessed keys based on frequency tracker
    void update_top_pinned(const std::unordered_map<Key, uint64_t> & freq_map, double top_fraction = 0.20) {
        if (freq_map.empty() || top_fraction <= 0.0) {
            pinned_.clear();
            return;
        }

        std::vector<std::pair<Key, uint64_t>> sorted(freq_map.begin(), freq_map.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
            return a.second > b.second;
        });

        size_t n_pin = (size_t) ((double) sorted.size() * top_fraction);
        n_pin = std::max<size_t>(1, std::min(n_pin, capacity_ / 2)); // cap pinned to at most half capacity

        pinned_.clear();
        for (size_t i = 0; i < n_pin && i < sorted.size(); ++i) {
            pinned_.insert(sorted[i].first);
        }
    }

    void clear() {
        t1_.clear();
        t2_.clear();
        b1_.clear();
        b2_.clear();
        entries_.clear();
        pinned_.clear();
        target_p_ = 0.0;
    }

private:
    struct Entry {
        Location location = Location::None;
        typename std::list<Key>::iterator it;
    };

    void remove_from_list(Location loc, typename std::list<Key>::iterator it) {
        switch (loc) {
        case Location::T1: t1_.erase(it); break;
        case Location::T2: t2_.erase(it); break;
        case Location::B1: b1_.erase(it); break;
        case Location::B2: b2_.erase(it); break;
        case Location::None: break;
        }
    }

    void replace(const Key & new_key, std::vector<Key> & evicted_resident) {
        bool in_b2 = false;
        auto it = entries_.find(new_key);
        if (it != entries_.end() && it->second.location == Location::B2) {
            in_b2 = true;
        }

        const size_t t1_sz = t1_.size();
        if (t1_sz > 0 && (((double) t1_sz > target_p_) || (in_b2 && ((double) t1_sz == target_p_)))) {
            // Evict from T1 to B1
            evict_lru_t1(evicted_resident, true /* move to B1 */);
        } else if (!t2_.empty()) {
            // Evict from T2 to B2
            evict_lru_t2(evicted_resident, true /* move to B2 */);
        } else if (!t1_.empty()) {
            evict_lru_t1(evicted_resident, true /* move to B1 */);
        }
    }

    void evict_lru_t1(std::vector<Key> & evicted_resident, bool move_to_b1) {
        // Find first non-pinned from back of T1
        auto it = t1_.end();
        while (it != t1_.begin()) {
            --it;
            const Key & k = *it;
            if (pinned_.find(k) == pinned_.end()) {
                evicted_resident.push_back(k);
                if (move_to_b1) {
                    b1_.push_front(k);
                    entries_[k].location = Location::B1;
                    entries_[k].it = b1_.begin();
                } else {
                    entries_.erase(k);
                }
                t1_.erase(it);
                return;
            }
        }
        // If all in T1 are pinned, fall back to first item anyway to preserve memory invariant
        if (!t1_.empty()) {
            const Key k = t1_.back();
            t1_.pop_back();
            evicted_resident.push_back(k);
            if (move_to_b1) {
                b1_.push_front(k);
                entries_[k].location = Location::B1;
                entries_[k].it = b1_.begin();
            } else {
                entries_.erase(k);
            }
        }
    }

    void evict_lru_t2(std::vector<Key> & evicted_resident, bool move_to_b2) {
        // Find first non-pinned from back of T2
        auto it = t2_.end();
        while (it != t2_.begin()) {
            --it;
            const Key & k = *it;
            if (pinned_.find(k) == pinned_.end()) {
                evicted_resident.push_back(k);
                if (move_to_b2) {
                    b2_.push_front(k);
                    entries_[k].location = Location::B2;
                    entries_[k].it = b2_.begin();
                } else {
                    entries_.erase(k);
                }
                t2_.erase(it);
                return;
            }
        }
        // If all in T2 are pinned, fall back to last item
        if (!t2_.empty()) {
            const Key k = t2_.back();
            t2_.pop_back();
            evicted_resident.push_back(k);
            if (move_to_b2) {
                b2_.push_front(k);
                entries_[k].location = Location::B2;
                entries_[k].it = b2_.begin();
            } else {
                entries_.erase(k);
            }
        }
    }

    void evict_ghost_b1() {
        if (!b1_.empty()) {
            const Key k = b1_.back();
            b1_.pop_back();
            entries_.erase(k);
        }
    }

    void evict_ghost_b2() {
        if (!b2_.empty()) {
            const Key k = b2_.back();
            b2_.pop_back();
            entries_.erase(k);
        }
    }

    size_t capacity_ = 0;
    double target_p_ = 0.0;

    std::list<Key> t1_; // Resident recency list (MRU at front)
    std::list<Key> t2_; // Resident frequency list (MRU at front)
    std::list<Key> b1_; // Ghost recency list (MRU at front)
    std::list<Key> b2_; // Ghost frequency list (MRU at front)

    std::unordered_map<Key, Entry> entries_;
    std::unordered_set<Key> pinned_; // Set of pinned keys
    mutable std::mutex mtx_;
};


} // namespace bmoe
