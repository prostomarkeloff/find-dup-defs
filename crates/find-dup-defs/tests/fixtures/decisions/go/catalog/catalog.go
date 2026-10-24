package catalog

import "example.com/shop/models"

// Offers lists what checkout would accept — the same rule, written as a filter.
func Offers(items []models.Item) []models.Item {
	var out []models.Item
	for _, item := range items {
		if item.Status == models.StatusOpen && item.Paid {
			out = append(out, item)
		}
	}
	return out
}

func Cheapest(items []models.Item) *models.Item {
	var best *models.Item
	for i := range items {
		if items[i].Price != nil && (best == nil || *items[i].Price < *best.Price) {
			best = &items[i]
		}
	}
	return best
}
