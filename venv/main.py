from tkinter import *

from PIL import ImageTk, Image

from algo.experiments import Experiment
from world.agent import Agent, ActionType
from world.cell import CellType, Cell
from world.pdworld import PDWorld


class GUI:
    main_window = Tk()
    main_window.title("Reinforcement Learning")
    main_window.geometry("250x500")
    view_world_button = None
    pd_world_window = None
    qTable_window = None
    pdWorld = None
    labels = []
    blocks = []
    qLabels = []
    qTable = {}
    agent = None

    def __init__(self, world, agent):
        self.pdWorld = world
        self.agent = agent

    def new_window(self, title):
        window = Toplevel(self.main_window)
        window.title(title)
        if title == "PD-World":
            self.pd_world_window = window
            self.pd_world_window.grid_columnconfigure(0, weight=1)
            self.pd_world_window.grid_columnconfigure(1, weight=1)
            self.pd_world_window.grid_columnconfigure(2, weight=1)
            self.pd_world_window.grid_columnconfigure(3, weight=1)
            self.pd_world_window.grid_columnconfigure(4, weight=1)
            self.pd_world_window.grid_columnconfigure(5, weight=4)
            self.pd_world_window.grid_rowconfigure(0, weight=1)
            self.pd_world_window.grid_rowconfigure(1, weight=1)
            self.pd_world_window.grid_rowconfigure(2, weight=0)
            self.pd_world_window.grid_rowconfigure(3, weight=1)
            self.pd_world_window.grid_rowconfigure(4, weight=0)
            self.pd_world_window.grid_rowconfigure(5, weight=1)
            self.pd_world_window.geometry("650x650")
            self.create_labels()
            self.updateAgentPosition(self.agent.agentPosition)
        if title == "Q-Table":
            self.qTable_window = window
            self.qTable_window.geometry("290x550")
            for c in range(1, len(self.pdWorld.cells) + 1):
                for o in range(1, len(self.agent.operators) + 1):
                    qLabel = Label(self.qTable_window, text='0', borderwidth=0, width=5, height=1, relief=SOLID,
                                   fg="white", bg="black")
                    qLabel.grid(row=c, column=o, sticky="", padx=2)
                    qLabel.columnconfigure(0, weight=1)
                    qLabel.rowconfigure(0, weight=1)
                    self.qLabels.append(qLabel)
                    self.qTable[(c, o)] = qLabel
                    Label(self.qTable_window, text=self.agent.operators[o - 1].type.name, font=("Helvetica", 7)).grid(
                        row=0,
                        column=o,
                        sticky="EW")
                    Label(self.qTable_window, text=self.pdWorld.cells[c - 1].position).grid(column=0, row=c,
                                                                                            sticky="NSEW")

    def create_labels(self):
        self.block_img = PhotoImage(file="money-bag.png")
        self.block_img = self.block_img.subsample(30)

        for i in self.pdWorld.cells:
            if i.type == CellType.PICKUP:
                frame = Frame(self.pd_world_window, background="blue")
                frame.grid(row=i.position[0], column=i.position[1], sticky="NSEW", pady=0, padx=0)

                label = Label(frame, text='(%s,%s)' % i.position, bd=1, fg="blue", background="blue",
                              font=("Helvetica", 12))
                label.grid(row=i.position[0], column=i.position[1], sticky='NSEW', pady=0, padx=0)

                for x in range(0, i.blocks):
                    block = Label(frame, fg="blue", bd=0, relief=RIDGE, bg="blue", compound=BOTTOM, height=15, width=15,
                                  image=self.block_img, anchor='center')
                    block.grid(row=2, column=x, pady=0, padx=0)
                    self.blocks.append((block, label))
                self.labels.append(label)

            elif i.type == CellType.DROPOFF:
                frame = Frame(self.pd_world_window, background="green")
                frame.grid(row=i.position[0], column=i.position[1], sticky="NSEW")

                label = Label(self.pd_world_window, text='(%s,%s)' % i.position, bd=1, fg="green", relief=GROOVE,
                              background="green", font=("Helvetica", 12))
                label.grid(row=i.position[0], column=i.position[1], sticky='NSEW')
                self.labels.append(label)

            else:
                frame = Frame(self.pd_world_window, background="black")
                frame.grid(row=i.position[0], column=i.position[1], sticky="NSEW")

                label = Label(self.pd_world_window, text='(%s,%s)' % i.position, bd=1, fg="black",
                              highlightthickness=50, relief=GROOVE, background="black", font=("Helvetica", 12))
                label.grid(row=i.position[0], column=i.position[1], sticky='NSEW')
                self.labels.append(label)

    def updateQTable(self, x, y, action, qValue):
        #self.qTable_window.update_idletasks()
        if (x, y) in self.qTable:
            state = self.checkState(x, y)
            self.qTable[state, action].configure(text=qValue)
        #self.qTable_window.update_idletasks()

    def updateAgentPosition(self, agentPos):
        self.pd_world_window.update_idletasks()
        for l in self.labels:
            if l.cget("text") == "(" + ','.join(map(str, agentPos.position)) + ")":
                l.config(image=self.agent.img)
                l.image = agent.img
            else:
                l.config(image='')
                l.image = ''
        self.pd_world_window.update_idletasks()

    def addBlock(self, cell):
        self.pd_world_window.update_idletasks()
        for b in self.blocks:
            if b[1].cget("text") == "(" + ','.join(map(str, cell.position)) + ")":
                b[1].config(compound=BOTTOM, height=15, width=15, image=self.block_img, anchor='w', justify=CENTER)
                b[1].image = self.block_img
                try:
                    self.pd_world_window.update_idletasks()
                except:
                    self.pd_world_window.mainloop()
                    self.pd_world_window.update_idletasks()
                    self.pd_world_window.update()

                break

    def removeBlock(self, cell):
        for b in self.blocks:
            if b[1].cget("text") == "(" + ','.join(map(str, cell.position)) + ")":
                b[0].destroy()
                #self.pd_world_window.mainloop()
                try:
                    self.pd_world_window.update_idletasks()
                    self.pd_world_window.update()
                except:
                    self.pd_world_window.mainloop()
                    self.pd_world_window.update_idletasks()
                    self.pd_world_window.update()

                break

    def create_pdworld(self):
        self.view_world_button = Button(self.main_window, text='View World', pady=10, width=25, background='#4d88ff',
                                        command=lambda: self.new_window("PD-World"))

        self.view_world_button.grid()
        self.view_world_button.place(relx=0.5, rely=0.1, anchor=CENTER)

        self.experiment1_button = Button(self.main_window, text='Experiment 1', pady=10, width=25, background='#ADFF2F',
                                         command=lambda: self.experiment2())
        self.experiment2_button = Button(self.main_window, text='Experiment 2', pady=10, width=25, background='#ADFF2F',
                                         command=lambda: self.experiment2())
        self.experiment3_button = Button(self.main_window, text='Experiment 3', pady=10, width=25, background='#ADFF2F',
                                         command=lambda: self.experiment2())
        self.experiment4_button = Button(self.main_window, text='Experiment 4', pady=10, width=25, background='#ADFF2F',
                                         command=lambda: self.experiment2())
        self.experiment5_button = Button(self.main_window, text='Experiment 5', pady=10, width=25, background='#ADFF2F',
                                         command=lambda: self.experiment2())
        self.experiment1_button.grid()
        self.experiment1_button.place(relx=0.5, rely=0.4, anchor=CENTER)
        self.experiment2_button.grid()
        self.experiment2_button.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.experiment3_button.grid()
        self.experiment3_button.place(relx=0.5, rely=0.6, anchor=CENTER)
        self.experiment4_button.grid()
        self.experiment4_button.place(relx=0.5, rely=0.7, anchor=CENTER)
        self.experiment5_button.grid()
        self.experiment5_button.place(relx=0.5, rely=0.8, anchor=CENTER)

    def create_qTable(self):
        self.view_qTable_button = Button(self.main_window, text='View Q-Table', pady=10, width=25, background='#4d88ff',
                                         command=lambda: self.new_window("Q-Table"))

        self.view_qTable_button.grid()
        self.view_qTable_button.place(relx=0.5, rely=.25, anchor=S)

    def checkState(self, x, y):
        x = x - 1
        total = x * 5 + y
        return total

    # def addBlocksToLabel(self, positionToAddBlock):
    #     for l in self.labels:
    #         if l.cget("text") == "("+','.join(map(str, positionToAddBlock))+ ")":
    #             l.config(image=Cell.block_img)

    def experiment2(self):
        (Experiment(self.agent)).experiment2()
        #self.updateAgentPosition(self.agent.agentPosition)

    def generate(self):
        self.main_window.mainloop()
        self.pd_world_window.mainloop()


world = PDWorld()
agent = Agent(world)
gui = GUI(world, agent)
agent.setGUI(gui)
gui.create_pdworld()
gui.create_qTable()
gui.generate()
