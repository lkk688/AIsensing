from tkinter import *
root=Tk()
# add widgets here

root.title('Hello Python TK')
#root.geometry("300x200+10+20")
#root.geometry("1600x800+100+20")
root.geometry("800x400+100+20")
#root.call("tk", "scaling", 1.50)  # MWT: This seemed to fix scaling issues in Windows.
# Verify it doesn't mess anything up on the Pi, detect os if so.
root.resizable(
    False, False
)  # the GUI is setup to be resizeable--and it all works great.
# EXCEPT when we do the blitting of the graphs ( to speed up redraw time).  I can't figure that out.  So, for now, I just made it un-resizeable...
root.minsize(700, 600)
root.mainloop()