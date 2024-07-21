from tkinter import *

root = Tk()
root.title("Hello World!")
root.geometry('300x300')
label = Label(root, text="hello")
label.grid()
entry = Entry(root, width=10)
entry.grid()


def click():
    value = entry.get()
    label.configure(text=value)


button = Button(root, text="Click", command=click)
button.grid(column=1, row=0)
root.mainloop()
