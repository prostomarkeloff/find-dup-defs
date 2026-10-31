package shop.catalog;

import java.util.ArrayList;
import java.util.List;

import shop.models.Item;
import shop.models.Status;

/** What checkout would accept — the same rule, written as a filter. */
public class Catalog {
    public static List<Item> offers(List<Item> items) {
        List<Item> out = new ArrayList<>();
        for (Item item : items) {
            if (Status.OPEN.code().equals(item.status()) && item.paid()) {
                out.add(item);
            }
        }
        return out;
    }

    public static Item cheapest(List<Item> items) {
        Item best = null;
        for (Item item : items) {
            if (item.price() != null && (best == null || item.price() < best.price())) {
                best = item;
            }
        }
        return best;
    }
}
