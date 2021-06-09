#pragma once

#include <vector>
#include <cassert>
#include <limits>
#include <numeric>

class union_find {
    std::vector<std::size_t> id;
    std::vector<std::size_t> sz;
    std::size_t cnt;

    public: 
    // Create an empty union find data structure with N isolated sets.
    void init(const std::size_t N)
    {
        id.resize(N);
        sz.resize(N);
        cnt = N;
        std::iota(id.begin(), id.end(), 0);
        std::fill(sz.begin(), sz.end(), 1);
    }

    union_find(const std::size_t N = 0) { init(N); }
    std::size_t size() const { assert(id.size() == sz.size()); return id.size(); }
    void reset() { init(this->size()); }

    // Return the id of component corresponding to object p.
    std::size_t find(std::size_t p) {
        assert(p < size());
        std::size_t root = p;
        while (root != id[root])
            root = id[root];
        while (p != root) {
            std::size_t newp = id[p];
            id[p] = root;
            p = newp;
        }
        return root;
    }
    // Replace sets containing x and y with their union.
    void merge(const std::size_t x, const std::size_t y) {
        const std::size_t i = find(x);
        const std::size_t j = find(y);
        if(i == j) return;

        // make smaller root point to larger one
        if(sz[i] < sz[j])    { 
            id[i] = j; 
            sz[j] += sz[i]; 
            sz[i] = 0;
        } else   { 
            id[j] = i; 
            sz[i] += sz[j]; 
            sz[j] = 0;
        }
        cnt--;
    }
    // Are objects x and y in the same set?
    bool connected(const std::size_t x, const std::size_t y) {
        return find(x) == find(y);
    }

    std::size_t no_elements(const std::size_t x) const
    {
        return sz[x];
    }

    std::size_t thread_safe_find(const std::size_t p) const {
        std::size_t root = p;
        while (root != id[root])
            root = id[root];
        return root;
    }
    bool thread_safe_connected(const std::size_t x, const std::size_t y) const {
        return thread_safe_find(x) == thread_safe_find(y);
    }
    // Return the number of disjoint sets.
    std::size_t count() const {
        return cnt;
    }

    std::vector<std::size_t> get_contiguous_ids()
    {
        std::vector<std::size_t> contiguous_ids(this->size());
        std::vector<std::size_t> id_mapping(this->size(), std::numeric_limits<std::size_t>::max());
        for(std::size_t i=0; i<this->size(); ++i) {
            std::size_t d = find(i);
            id_mapping[d] = 1; 
        }
        std::size_t next_id = 0;
        for(std::size_t d=0; d<this->size(); ++d) {
            if(id_mapping[d] == 1) {
                id_mapping[d] = next_id;
                ++next_id;
            }
        }

        for(std::size_t i=0; i<this->size(); ++i) {
            std::size_t d = find(i);
            assert(id_mapping[d] != std::numeric_limits<std::size_t>::max());
            contiguous_ids[i] = id_mapping[d];
        }
        return id_mapping;
    }
};
