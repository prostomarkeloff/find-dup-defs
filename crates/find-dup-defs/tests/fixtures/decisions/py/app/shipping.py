from app.models import Carrier

def check_0(parcel):
    if parcel is None:
        return False
    if parcel.weight_0 > 1:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_0

def check_1(parcel):
    if parcel is None:
        return False
    if parcel.weight_1 > 2:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_1

def check_2(parcel):
    if parcel is None:
        return False
    if parcel.weight_2 > 3:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_2

def check_3(parcel):
    if parcel is None:
        return False
    if parcel.weight_3 > 4:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_3

def check_4(parcel):
    if parcel is None:
        return False
    if parcel.weight_4 > 5:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_4

def check_5(parcel):
    if parcel is None:
        return False
    if parcel.weight_5 > 6:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_5

def check_6(parcel):
    if parcel is None:
        return False
    if parcel.weight_6 > 7:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_6

def check_7(parcel):
    if parcel is None:
        return False
    if parcel.weight_7 > 8:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_0

def check_8(parcel):
    if parcel is None:
        return False
    if parcel.weight_8 > 9:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_1

def check_9(parcel):
    if parcel is None:
        return False
    if parcel.weight_9 > 10:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_2

def check_10(parcel):
    if parcel is None:
        return False
    if parcel.weight_10 > 11:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_3

def check_11(parcel):
    if parcel is None:
        return False
    if parcel.weight_11 > 12:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_4

def check_12(parcel):
    if parcel is None:
        return False
    if parcel.weight_12 > 13:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_5

def check_13(parcel):
    if parcel is None:
        return False
    if parcel.weight_13 > 14:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_6

def check_14(parcel):
    if parcel is None:
        return False
    if parcel.weight_14 > 15:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_0

def check_15(parcel):
    if parcel is None:
        return False
    if parcel.weight_15 > 16:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_1

def check_16(parcel):
    if parcel is None:
        return False
    if parcel.weight_16 > 17:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_2

def check_17(parcel):
    if parcel is None:
        return False
    if parcel.weight_17 > 18:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_3

def check_18(parcel):
    if parcel is None:
        return False
    if parcel.weight_18 > 19:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_4

def check_19(parcel):
    if parcel is None:
        return False
    if parcel.weight_19 > 20:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_5

def check_20(parcel):
    if parcel is None:
        return False
    if parcel.weight_20 > 21:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_6

def check_21(parcel):
    if parcel is None:
        return False
    if parcel.weight_21 > 22:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_0

def check_22(parcel):
    if parcel is None:
        return False
    if parcel.weight_22 > 23:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_1

def check_23(parcel):
    if parcel is None:
        return False
    if parcel.weight_23 > 24:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_2

def check_24(parcel):
    if parcel is None:
        return False
    if parcel.weight_24 > 25:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_3

def check_25(parcel):
    if parcel is None:
        return False
    if parcel.weight_25 > 26:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_4

def check_26(parcel):
    if parcel is None:
        return False
    if parcel.weight_26 > 27:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_5

def check_27(parcel):
    if parcel is None:
        return False
    if parcel.weight_27 > 28:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_6

def check_28(parcel):
    if parcel is None:
        return False
    if parcel.weight_28 > 29:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_0

def check_29(parcel):
    if parcel is None:
        return False
    if parcel.weight_29 > 30:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_1

def check_30(parcel):
    if parcel is None:
        return False
    if parcel.weight_30 > 31:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_2

def check_31(parcel):
    if parcel is None:
        return False
    if parcel.weight_31 > 32:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_3

def check_32(parcel):
    if parcel is None:
        return False
    if parcel.weight_32 > 33:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_4

def check_33(parcel):
    if parcel is None:
        return False
    if parcel.weight_33 > 34:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_5

def check_34(parcel):
    if parcel is None:
        return False
    if parcel.weight_34 > 35:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_6

def check_35(parcel):
    if parcel is None:
        return False
    if parcel.weight_35 > 36:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_0

def check_36(parcel):
    if parcel is None:
        return False
    if parcel.weight_36 > 37:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_1

def check_37(parcel):
    if parcel is None:
        return False
    if parcel.weight_37 > 38:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_2

def check_38(parcel):
    if parcel is None:
        return False
    if parcel.weight_38 > 39:
        return False
    return parcel.carrier == Carrier.COURIER or parcel.flag_3

def check_39(parcel):
    if parcel is None:
        return False
    if parcel.weight_39 > 40:
        return False
    return parcel.carrier == Carrier.POST or parcel.flag_4
