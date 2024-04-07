from world.cell import Cell, CellType

"""Custom Pick-Up Drop-off World"""""


class PDWorld:
    # startCell = (1,5)
    startCell = Cell(CellType.REGULAR, (1, 5))
    cells = []

    def __init__(self):
        self.cells = []
        for x in range(1, 6):
            for y in range(1, 6):
                if x == y and (x % 2 == 1):
                    self.cells.append(Cell(CellType.PICKUP, (x, y)))
                elif x == 2 and y == 5:
                    self.cells.append(Cell(CellType.DROPOFF, (x, y)))
                elif x == 5 and y == 1:
                    self.cells.append(Cell(CellType.DROPOFF, (x, y)))
                elif x == 5 and y == 3:
                    self.cells.append(Cell(CellType.DROPOFF, (x, y)))
                else:
                    self.cells.append(Cell(CellType.REGULAR, (x, y)))

    def isInTerminalState(self):
        pickupDone = False
        dropoffDone = False
        for c in self.cells:
            if c.type == CellType.PICKUP and c.blocks == 0:
                pickupDone = True
        for c in self.cells:
            if c.type == CellType.DROPOFF and c.blocks == 5:
                dropoffDone = True
        return pickupDone and dropoffDone

    def getCell(self, x, y):
        for c in self.cells:
            if c.position[0] == x and c.position[1] == y:
                return c
        print("Could not find a cell with coordinates: " + "(", x, ",", y, ")")
