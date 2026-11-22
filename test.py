import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

def onclick(event):
    if event.inaxes:
        print(f"Clicked at x={event.xdata}, y={event.ydata}")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()