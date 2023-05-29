#include "../common/common.h"

class Segment_tree_zkw {
    private:
        int tn;
        int size;
    public:
        std::vector<int> t;
        std::vector<int> num;
        Segment_tree_zkw(int N){
            size = (N+1)<<2;
            this->t = std::vector<int>(size+1 , INT_MAX);
            this->num = std::vector<int>(size+1 , 0);
        }
        void init(int N) {
            for(int i = 0; i < size; i++){
                this->t[i] = INT_MAX;
                this->num[i] = 0;
            }
            
            tn = 1;
            while(tn < N) tn <<= 1;
            --tn;
            for (int i = 1; i <= N; ++i)
                this->num[i + tn] = i;
        }

        void change(int p, const int &k) {
            p += tn; this->t[p] = k; p >>= 1;
            while (p) {
                if (this->t[p<<1] < this->t[p<<1|1]) {
                    this->t[p] = this->t[p<<1];
                    this->num[p] = this->num[p<<1];
                }
                else {
                    this->t[p] = this->t[p<<1|1];
                    this->num[p] = this->num[p<<1|1];
                }
                p >>= 1;
            }
        }
};