

typedef struct fi_cq{
    void (*func)(int, int);
    int rank;
} fi_cq_t;

fi_cq_t cq_obj;
