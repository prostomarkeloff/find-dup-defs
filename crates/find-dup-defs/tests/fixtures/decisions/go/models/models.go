package models

type Status string

const (
	StatusOpen   Status = "open"
	StatusHeld   Status = "held"
	StatusClosed Status = "closed"
)

type Carrier string

const (
	CarrierPost    Carrier = "post"
	CarrierCourier Carrier = "courier"
)

type Order struct {
	ShopID   int
	Status   Status
	Paid     bool
	PlacedAt int64
	Total    int
	Number   *string
	Lines    []Line
}

type Line struct {
	Price    int
	Quantity int
}

type Item struct {
	Status Status
	Paid   bool
	Price  *int
}
