package shop.orders;

import shop.models.Order;
import shop.models.Status;

/** How checkout refuses an order that is not open and paid. */
public class Orders {
    public static void refuse(Order order) {
        if (!order.status().equals(Status.OPEN.code())) {
            throw new IllegalStateException("order closed");
        }
        if (!order.paid()) {
            throw new IllegalStateException("order unpaid");
        }
    }

    public static int total(Order order) {
        int sum = 0;
        for (var line : order.lines()) {
            if (line.quantity() > 0) {
                sum += line.price();
            }
        }
        return sum;
    }

    public static String label(Order order) {
        if (order.number() != null) {
            return order.number();
        }
        return "draft";
    }
}
