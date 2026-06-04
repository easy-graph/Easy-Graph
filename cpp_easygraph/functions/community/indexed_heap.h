#pragma once

#include <vector>
#include <algorithm>

class IndexedMaxHeap {
public:
    struct HeapEntry {
        double dq;
        int i;
        int j;
    };

private:
    std::vector<HeapEntry> heap;
    std::vector<std::vector<int>> pair_to_idx;
    size_t heap_size;
    int capacity_n;

    static inline long long make_key(int i, int j) {
        int a = std::min(i, j);
        int b = std::max(i, j);
        return ((long long)a << 32) | (unsigned int)b;
    }

public:
    IndexedMaxHeap(size_t capacity = 0) : heap_size(0), capacity_n((int)capacity) {
        heap.reserve(capacity);
        if (capacity > 0) {
            pair_to_idx.assign(capacity, std::vector<int>(capacity, -1));
        }
    }

    bool empty() const { return heap_size == 0; }
    size_t size() const { return heap_size; }

    int get_index(int i, int j) const {
        int a = std::min(i, j);
        int b = std::max(i, j);
        if (a < 0 || b < 0 || a >= capacity_n || b >= capacity_n) return -1;
        size_t idx = (size_t)pair_to_idx[a][b];
        if (idx >= heap.size()) return -1;
        if (heap[idx].i == a && heap[idx].j == b) {
            return (int)idx;
        }
        return -1;
    }

    void push(double dq, int i, int j) {
        int a = std::min(i, j);
        int b = std::max(i, j);
        if (a < 0 || b < 0 || a >= capacity_n || b >= capacity_n) return;

        size_t existing = (size_t)pair_to_idx[a][b];
        if (existing < heap.size() && heap[existing].i == a && heap[existing].j == b) {
            update(dq, i, j);
            return;
        }

        HeapEntry entry = {dq, a, b};
        heap.push_back(entry);
        size_t idx = heap.size() - 1;
        pair_to_idx[a][b] = (int)idx;
        sift_up(idx);
        heap_size++;
    }

    void update(double dq, int i, int j) {
        int a = std::min(i, j);
        int b = std::max(i, j);
        if (a < 0 || b < 0 || a >= capacity_n || b >= capacity_n) return;

        size_t idx = (size_t)pair_to_idx[a][b];
        if (idx >= heap.size()) return;
        if (heap[idx].i != a || heap[idx].j != b) return;
        if (heap[idx].dq == dq) return;

        bool is_increase = dq > heap[idx].dq;
        heap[idx].dq = dq;

        if (is_increase) {
            sift_up(idx);
        } else {
            sift_down(idx);
        }
    }

    void remove(int i, int j) {
        int a = std::min(i, j);
        int b = std::max(i, j);
        if (a < 0 || b < 0 || a >= capacity_n || b >= capacity_n) return;

        size_t idx = (size_t)pair_to_idx[a][b];
        if (idx >= heap.size()) return;
        if (heap[idx].i != a || heap[idx].j != b) return;

        pair_to_idx[a][b] = -1;

        if (idx == heap.size() - 1) {
            heap.pop_back();
        } else {
            heap[idx] = heap.back();
            int last_a = heap[idx].i;
            int last_b = heap[idx].j;
            if (last_a >= 0 && last_a < capacity_n && last_b >= 0 && last_b < capacity_n) {
                pair_to_idx[last_a][last_b] = (int)idx;
            }
            heap.pop_back();
            sift_up(idx);
            sift_down(idx);
        }
        if (heap_size > 0) heap_size--;
    }

    HeapEntry pop() {
        HeapEntry result = heap[0];
        pair_to_idx[result.i][result.j] = -1;

        if (heap.size() > 1) {
            heap[0] = heap.back();
            int new_a = heap[0].i;
            int new_b = heap[0].j;
            if (new_a >= 0 && new_a < capacity_n && new_b >= 0 && new_b < capacity_n) {
                pair_to_idx[new_a][new_b] = 0;
            }
            heap.pop_back();
            sift_down(0);
        } else {
            heap.pop_back();
        }
        if (heap_size > 0) heap_size--;

        return result;
    }

    const HeapEntry& top() const { return heap[0]; }

private:
    void sift_up(size_t idx) {
        while (idx > 0) {
            size_t parent = (idx - 1) >> 1;
            if (heap[parent].dq >= heap[idx].dq) break;

            std::swap(heap[parent], heap[idx]);

            int p_i = heap[parent].i, p_j = heap[parent].j;
            int c_i = heap[idx].i, c_j = heap[idx].j;

            if (p_i >= 0 && p_i < capacity_n && p_j >= 0 && p_j < capacity_n) {
                pair_to_idx[p_i][p_j] = (int)parent;
            }
            if (c_i >= 0 && c_i < capacity_n && c_j >= 0 && c_j < capacity_n) {
                pair_to_idx[c_i][c_j] = (int)idx;
            }

            idx = parent;
        }
    }

    void sift_down(size_t idx) {
        size_t n = heap.size();
        while (true) {
            size_t largest = idx;
            size_t left = idx * 2 + 1;
            size_t right = idx * 2 + 2;

            if (left < n && heap[left].dq > heap[largest].dq) {
                largest = left;
            }
            if (right < n && heap[right].dq > heap[largest].dq) {
                largest = right;
            }

            if (largest == idx) break;

            std::swap(heap[largest], heap[idx]);

            int l_i = heap[largest].i, l_j = heap[largest].j;
            int c_i = heap[idx].i, c_j = heap[idx].j;

            if (l_i >= 0 && l_i < capacity_n && l_j >= 0 && l_j < capacity_n) {
                pair_to_idx[l_i][l_j] = (int)largest;
            }
            if (c_i >= 0 && c_i < capacity_n && c_j >= 0 && c_j < capacity_n) {
                pair_to_idx[c_i][c_j] = (int)idx;
            }

            idx = largest;
        }
    }
};
