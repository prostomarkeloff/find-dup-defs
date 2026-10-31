package shop.shipping;

import shop.models.Carrier;
import shop.models.Parcel;

/** What shipping charges for a parcel — the same rule spelled many ways. */
public class Shipping {
    public static int fee(Parcel parcel) {
        if (parcel.carrier() == Carrier.POST) {
            return post(parcel);
        }
        if (parcel.carrier() == Carrier.COURIER) {
            return courier(parcel);
        }
        return 0;
    }

    public static int post(Parcel parcel) {
        if (parcel.weight() == null) {
            return 0;
        }
        return parcel.weight() * 5;
    }

    public static int courier(Parcel parcel) {
        if (parcel.weight() == null) {
            return 120;
        }
        return parcel.weight() * 10;
    }

    public static boolean check0(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 1) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check1(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 2) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check2(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 3) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check3(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 4) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check4(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 5) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check5(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 6) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check6(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 7) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check7(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 8) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check8(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 9) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check9(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 10) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check10(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 11) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check11(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 12) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check12(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 13) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check13(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 14) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check14(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 15) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check15(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 16) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check16(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 17) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check17(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 18) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check18(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 19) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check19(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 20) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check20(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 21) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check21(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 22) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check22(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 23) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check23(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 24) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check24(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 25) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check25(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 26) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check26(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 27) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check27(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 28) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check28(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 29) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check29(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 30) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check30(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 31) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check31(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 32) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check32(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 33) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check33(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 34) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check34(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 35) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check35(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 36) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check36(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 37) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check37(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 38) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }

    public static boolean check38(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 39) {
            return false;
        }
        return parcel.carrier() == Carrier.COURIER || parcel.paid();
    }

    public static boolean check39(Parcel parcel) {
        if (parcel.weight() == null) {
            return false;
        }
        if (parcel.weight() > 40) {
            return false;
        }
        return parcel.carrier() == Carrier.POST || parcel.paid();
    }
}
