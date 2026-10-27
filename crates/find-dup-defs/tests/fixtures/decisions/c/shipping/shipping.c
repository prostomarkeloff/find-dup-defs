#include <stddef.h>
#include "../models/models.h"

struct parcel {
    enum carrier carrier;
    int weight0;
    int weight1;
    int weight2;
    int weight3;
    int weight4;
    int weight5;
    int weight6;
    int weight7;
    int weight8;
    int weight9;
    int weight10;
    int weight11;
    int weight12;
    int weight13;
    int weight14;
    int weight15;
    int weight16;
    int weight17;
    int weight18;
    int weight19;
    int weight20;
    int weight21;
    int weight22;
    int weight23;
    int weight24;
    int weight25;
    int weight26;
    int weight27;
    int weight28;
    int weight29;
    int weight30;
    int weight31;
    int weight32;
    int weight33;
    int weight34;
    int weight35;
    int weight36;
    int weight37;
    int weight38;
    int weight39;
    int flag0;
    int flag1;
    int flag2;
    int flag3;
    int flag4;
    int flag5;
    int flag6;
};

int check0(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight0 > 1)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag0;
}

int check1(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight1 > 2)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag1;
}

int check2(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight2 > 3)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag2;
}

int check3(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight3 > 4)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag3;
}

int check4(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight4 > 5)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag4;
}

int check5(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight5 > 6)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag5;
}

int check6(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight6 > 7)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag6;
}

int check7(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight7 > 8)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag0;
}

int check8(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight8 > 9)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag1;
}

int check9(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight9 > 10)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag2;
}

int check10(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight10 > 11)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag3;
}

int check11(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight11 > 12)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag4;
}

int check12(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight12 > 13)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag5;
}

int check13(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight13 > 14)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag6;
}

int check14(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight14 > 15)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag0;
}

int check15(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight15 > 16)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag1;
}

int check16(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight16 > 17)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag2;
}

int check17(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight17 > 18)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag3;
}

int check18(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight18 > 19)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag4;
}

int check19(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight19 > 20)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag5;
}

int check20(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight20 > 21)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag6;
}

int check21(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight21 > 22)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag0;
}

int check22(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight22 > 23)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag1;
}

int check23(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight23 > 24)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag2;
}

int check24(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight24 > 25)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag3;
}

int check25(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight25 > 26)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag4;
}

int check26(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight26 > 27)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag5;
}

int check27(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight27 > 28)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag6;
}

int check28(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight28 > 29)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag0;
}

int check29(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight29 > 30)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag1;
}

int check30(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight30 > 31)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag2;
}

int check31(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight31 > 32)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag3;
}

int check32(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight32 > 33)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag4;
}

int check33(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight33 > 34)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag5;
}

int check34(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight34 > 35)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag6;
}

int check35(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight35 > 36)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag0;
}

int check36(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight36 > 37)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag1;
}

int check37(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight37 > 38)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag2;
}

int check38(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight38 > 39)
        return 0;
    return parcel->carrier == CARRIER_COURIER || parcel->flag3;
}

int check39(const struct parcel *parcel)
{
    if (parcel == NULL)
        return 0;
    if (parcel->weight39 > 40)
        return 0;
    return parcel->carrier == CARRIER_POST || parcel->flag4;
}
