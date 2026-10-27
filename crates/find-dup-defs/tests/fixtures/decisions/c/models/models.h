#ifndef MODELS_H
#define MODELS_H

enum status { STATUS_OPEN, STATUS_HELD, STATUS_CLOSED };

enum carrier { CARRIER_POST, CARRIER_COURIER };

struct line {
    int price;
    int quantity;
};

struct order {
    int shop_id;
    enum status status;
    int paid;
    long placed_at;
    int total;
    const char *number;
    struct line *lines;
    int line_count;
};

struct item {
    enum status status;
    int paid;
    const int *price;
};

#endif
