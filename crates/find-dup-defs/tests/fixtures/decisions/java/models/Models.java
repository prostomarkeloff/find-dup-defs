package shop.models;

import java.util.List;

public enum Status {
    OPEN("open"),
    HELD("held"),
    CLOSED("closed");

    private final String code;

    Status(String code) {
        this.code = code;
    }
}

public enum Carrier {
    POST("post"),
    COURIER("courier");

    private final String code;

    Carrier(String code) {
        this.code = code;
    }
}

public record Line(int price, int quantity) {
}

public record Order(
        int shopId,
        String status,
        boolean paid,
        int placedAt,
        int total,
        String number,
        List<Line> lines) {
}

public record Item(String status, boolean paid, Integer price) {
}

public record Parcel(Carrier carrier, String status, boolean paid, Integer weight) {
}
