import tkinter as tk
from gui import LiteratureReviewGUI
from datamining import PDFDataMiner
if __name__ == "__main__":
   root = tk.Tk()
   app = LiteratureReviewGUI(root)
   root.mainloop()