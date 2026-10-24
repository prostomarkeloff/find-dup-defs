package orders

import (
	"errors"

	"example.com/shop/models"
)

var (
	ErrClosed = errors.New("order closed")
	ErrUnpaid = errors.New("order unpaid")
)

// Refuse is how checkout refuses an order that is not open and paid.
func Refuse(order *models.Order) error {
	if order.Status != models.StatusOpen {
		return ErrClosed
	}
	if !order.Paid {
		return ErrUnpaid
	}
	return nil
}

func Total(order *models.Order) int {
	sum := 0
	for _, line := range order.Lines {
		if line.Quantity > 0 {
			sum += line.Price
		}
	}
	return sum
}

func Label(order *models.Order) string {
	if order.Number != nil {
		return *order.Number
	}
	return "draft"
}
