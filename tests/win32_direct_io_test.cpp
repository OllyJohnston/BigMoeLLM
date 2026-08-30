// Unit tests for Windows Direct I/O (NO_BUFFERING + OVERLAPPED) and Pinned Allocations.
#include "platform_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

using namespace bmoe::pio;

static int failures = 0;

static void test_pinned_allocation() {
    size_t sz = 16 * 1024 * 1024; // 16 MB test pool
    PinnedAlloc alloc;
    bool ok = pinned_alloc(sz, &alloc);
    if (!ok || !alloc.base) {
        std::printf("[FAIL] pinned_alloc failed for %zu bytes\n", sz);
        ++failures;
        return;
    }

    // Verify 4096-byte alignment
    uintptr_t addr = (uintptr_t) alloc.base;
    if ((addr & 4095) != 0) {
        std::printf("[FAIL] pinned_alloc base pointer %p is not 4096-byte aligned\n", alloc.base);
        ++failures;
        pinned_free(&alloc);
        return;
    }

    // Write and read test
    uint8_t * p = (uint8_t *) alloc.base;
    for (size_t i = 0; i < sz; i += 4096) {
        p[i] = (uint8_t) (i & 0xFF);
    }
    for (size_t i = 0; i < sz; i += 4096) {
        if (p[i] != (uint8_t) (i & 0xFF)) {
            std::printf("[FAIL] Memory verification failed at offset %zu\n", i);
            ++failures;
            pinned_free(&alloc);
            return;
        }
    }

    pinned_free(&alloc);
    if (alloc.base != nullptr) {
        std::printf("[FAIL] pinned_free should reset base pointer to null\n");
        ++failures;
        return;
    }

    std::printf("[PASS] Pinned allocation and 4096-byte alignment\n");
}

static void test_direct_io_read() {
    // Create a temporary test file aligned to 4096 bytes
    const char * tmp_filename = "test_direct_io_temp.bin";
    const size_t test_size = 64 * 1024; // 64 KB (16 sectors)

    std::vector<uint8_t> test_data(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        test_data[i] = (uint8_t) ((i * 31 + 7) & 0xFF);
    }

    FILE * f = std::fopen(tmp_filename, "wb");
    if (!f) {
        std::printf("[FAIL] Could not create temp file for direct I/O test\n");
        ++failures;
        return;
    }
    std::fwrite(test_data.data(), 1, test_size, f);
    std::fclose(f);

    // Open with direct = true (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
    fd_t fd = open_read(tmp_filename, true);
    if (!fd_ok(fd)) {
        std::printf("[FAIL] open_read direct failed for %s\n", tmp_filename);
        ++failures;
        std::remove(tmp_filename);
        return;
    }

    uint64_t fsz = file_size(fd);
    if (fsz != test_size) {
        std::printf("[FAIL] file_size returned %llu, expected %zu\n", (unsigned long long) fsz, test_size);
        ++failures;
    }

    // Allocate 4096-byte aligned read buffer
    void * buf = alloc_aligned(4096, test_size);
    if (!buf) {
        std::printf("[FAIL] alloc_aligned failed\n");
        ++failures;
        close_fd(fd);
        std::remove(tmp_filename);
        return;
    }

    // Read using pread_at (asynchronous Overlapped Direct I/O under the hood)
    long long bytes_read = pread_at(fd, buf, test_size, 0);
    if (bytes_read != (long long) test_size) {
        std::printf("[FAIL] pread_at returned %lld bytes, expected %zu\n", bytes_read, test_size);
        ++failures;
    } else {
        if (std::memcmp(buf, test_data.data(), test_size) != 0) {
            std::printf("[FAIL] Read data did not match written pattern\n");
            ++failures;
        } else {
            std::printf("[PASS] Windows Direct I/O async read byte verification\n");
        }
    }

    aligned_free(buf);
    close_fd(fd);
    std::remove(tmp_filename);
}

int main() {
    test_pinned_allocation();
    test_direct_io_read();

    if (failures == 0) {
        std::printf("All Direct I/O tests passed successfully.\n");
        return 0;
    } else {
        std::printf("%d Direct I/O test(s) failed.\n", failures);
        return 1;
    }
}
