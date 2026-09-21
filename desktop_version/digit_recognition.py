"""Desktop drawing application. Created 2026-09-21 KST."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw
from recognition import Recognizer

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.recognizer = Recognizer()
        root.title('MNIST | Handwritten Digit Studio')
        root.resizable(False, False)
        root.configure(bg='#eef2f7')
        frame = ttk.Frame(root, padding=28)
        frame.pack()
        ttk.Label(frame, text='Handwritten Digit Studio', font=('Segoe UI', 22, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Draw one digit from 0 to 9. Then click Recognize.').pack(anchor='w', pady=(8, 18))
        self.canvas = tk.Canvas(frame, width=320, height=320, bg='white', highlightthickness=1, highlightbackground='#b7c5da')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.start)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.finish)
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=16)
        ttk.Button(controls, text='Clear', command=self.clear).pack(side='left')
        ttk.Button(controls, text='Recognize', command=self.recognize).pack(side='right')
        self.live = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='Auto recognize', variable=self.live).pack(side='left', padx=12)
        self.result = ttk.Label(frame, text='', font=('Segoe UI', 18, 'bold'))
        self.result.pack(pady=6)
        self.details = ttk.Label(frame, text='')
        self.details.pack()
        ttk.Label(frame, text='Confidence is a model score, not a guarantee.').pack(pady=(14, 0))
        self.pending = None
        self.last = None
        self.clear()

    def clear(self):
        if self.pending:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.canvas.delete('all')
        self.image = Image.new('L', (320, 320), 255)
        self.pen = ImageDraw.Draw(self.image)
        self.last = None
        self.result.config(text='Ready to draw')
        self.details.config(text='')

    def start(self, event):
        if self.pending:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.last = (event.x, event.y)
        self.draw(event)

    def draw(self, event):
        if self.last is None:
            return
        x, y = max(0, min(319, event.x)), max(0, min(319, event.y))
        self.canvas.create_line(*self.last, x, y, width=20, fill='#111111', capstyle=tk.ROUND, smooth=True)
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='#111111', outline='')
        self.pen.line([self.last, (x, y)], fill=17, width=20)
        self.pen.ellipse((x-10, y-10, x+10, y+10), fill=17)
        self.last = (x, y)
        self.result.config(text='Drawing...')
        self.details.config(text='')

    def finish(self, event):
        self.last = None
        if self.live.get():
            self.pending = self.root.after(350, self.recognize)

    def recognize(self):
        if self.pending:
            self.root.after_cancel(self.pending)
            self.pending = None
        try:
            result = self.recognizer.predict(self.image)
            self.result.config(text=f"Digit {result['digit']}   |   {result['confidence']:.1%}")
            top = sorted(enumerate(result['probabilities']), key=lambda p: p[1], reverse=True)[:3]
            self.details.config(text='   '.join(f'{digit}: {score:.1%}' for digit, score in top))
        except ValueError as exc:
            self.result.config(text=str(exc))

def main():
    root = tk.Tk()
    try:
        DigitApp(root)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror('Startup error', str(exc))
        root.destroy()
        return
    root.mainloop()

if __name__ == '__main__':
    main()
