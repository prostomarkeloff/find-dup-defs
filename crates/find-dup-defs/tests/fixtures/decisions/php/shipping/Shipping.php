<?php

namespace Shop\Shipping;

use Shop\Models\Carrier;

function check0(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight0 > 1) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag0;
}

function check1(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight1 > 2) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag1;
}

function check2(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight2 > 3) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag2;
}

function check3(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight3 > 4) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag3;
}

function check4(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight4 > 5) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag4;
}

function check5(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight5 > 6) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag5;
}

function check6(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight6 > 7) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag6;
}

function check7(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight7 > 8) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag0;
}

function check8(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight8 > 9) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag1;
}

function check9(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight9 > 10) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag2;
}

function check10(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight10 > 11) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag3;
}

function check11(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight11 > 12) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag4;
}

function check12(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight12 > 13) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag5;
}

function check13(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight13 > 14) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag6;
}

function check14(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight14 > 15) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag0;
}

function check15(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight15 > 16) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag1;
}

function check16(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight16 > 17) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag2;
}

function check17(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight17 > 18) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag3;
}

function check18(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight18 > 19) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag4;
}

function check19(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight19 > 20) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag5;
}

function check20(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight20 > 21) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag6;
}

function check21(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight21 > 22) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag0;
}

function check22(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight22 > 23) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag1;
}

function check23(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight23 > 24) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag2;
}

function check24(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight24 > 25) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag3;
}

function check25(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight25 > 26) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag4;
}

function check26(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight26 > 27) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag5;
}

function check27(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight27 > 28) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag6;
}

function check28(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight28 > 29) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag0;
}

function check29(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight29 > 30) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag1;
}

function check30(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight30 > 31) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag2;
}

function check31(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight31 > 32) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag3;
}

function check32(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight32 > 33) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag4;
}

function check33(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight33 > 34) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag5;
}

function check34(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight34 > 35) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag6;
}

function check35(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight35 > 36) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag0;
}

function check36(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight36 > 37) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag1;
}

function check37(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight37 > 38) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag2;
}

function check38(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight38 > 39) {
        return false;
    }
    return $parcel->carrier === Carrier::COURIER || $parcel->flag3;
}

function check39(?object $parcel): bool
{
    if ($parcel === null) {
        return false;
    }
    if ($parcel->weight39 > 40) {
        return false;
    }
    return $parcel->carrier === Carrier::POST || $parcel->flag4;
}
