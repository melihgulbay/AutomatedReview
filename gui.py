import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json
from threading import Thread
import os
from llm import LiteratureReviewLLM
from datamining import PDFDataMiner
from PIL import Image, ImageTk

class LiteratureReviewGUI:
   def __init__(self, root):
       self.root = root
       self.root.title("Automated Literature Review")
       self.root.geometry("1000x800")
       
       # Initialize LLM
       self.llm = LiteratureReviewLLM()
       
       # Initialize API key
       self.api_key = ""
       self.load_api_key()
       
       self.data_miner = PDFDataMiner()
       
       self.setup_ui()
       
   def load_api_key(self):
       try:
           with open('config.json', 'r') as f:
               config = json.load(f)
               self.api_key = config.get('api_key', '')
       except FileNotFoundError:
           pass
           
   def save_api_key(self):
       with open('config.json', 'w') as f:
           json.dump({'api_key': self.api_key}, f)
           
   def setup_ui(self):
       # Create main canvas with scrollbar
       main_canvas = tk.Canvas(self.root)
       scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
       scrollable_frame = ttk.Frame(main_canvas)

       scrollable_frame.bind(
           "<Configure>",
           lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
       )

       main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
       main_canvas.configure(yscrollcommand=scrollbar.set)

       # API Key Frame
       api_frame = ttk.LabelFrame(scrollable_frame, text="API Configuration", padding=10)
       api_frame.pack(fill="x", padx=10, pady=5)
       
       ttk.Label(api_frame, text="Gemini API Key:").pack(side="left")
       self.api_entry = ttk.Entry(api_frame, width=50, show="*")
       self.api_entry.pack(side="left", padx=5)
       self.api_entry.insert(0, self.api_key)
       
       ttk.Button(api_frame, text="Save API Key", command=self.update_api_key).pack(side="left", padx=5)
       
       # PDF Upload Frame
       upload_frame = ttk.LabelFrame(scrollable_frame, text="PDF Upload", padding=10)
       upload_frame.pack(fill="x", padx=10, pady=5)
       
       self.file_label = ttk.Label(upload_frame, text="No file selected")
       self.file_label.pack(side="left", fill="x", expand=True)
       
       ttk.Button(upload_frame, text="Upload PDF", command=self.upload_pdf).pack(side="right")
       
       # Analysis Options Frame
       options_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Options", padding=10)
       options_frame.pack(fill="x", padx=10, pady=5)
       
       self.include_summary = tk.BooleanVar(value=True)
       ttk.Checkbutton(options_frame, text="Include Summary", variable=self.include_summary).pack(side="left")
       
       self.include_key_findings = tk.BooleanVar(value=True)
       ttk.Checkbutton(options_frame, text="Include Key Findings", variable=self.include_key_findings).pack(side="left")
       
       self.include_methodology = tk.BooleanVar(value=True)
       ttk.Checkbutton(options_frame, text="Include Methodology", variable=self.include_methodology).pack(side="left")
       
       # Output Frame
       output_frame = ttk.LabelFrame(scrollable_frame, text="Analysis Results", padding=10)
       output_frame.pack(fill="both", expand=True, padx=10, pady=5)
       
       self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20)
       self.output_text.pack(fill="both", expand=True)
       
       # Data Mining Results Frame
       mining_frame = ttk.LabelFrame(scrollable_frame, text="Data Mining Results", padding=10)
       mining_frame.pack(fill="x", padx=10, pady=5)

       # Text results
       self.mining_text = scrolledtext.ScrolledText(mining_frame, wrap=tk.WORD, height=10)
       self.mining_text.pack(fill="x", padx=5, pady=5)

       # Visualizations frame
       viz_frame = ttk.LabelFrame(scrollable_frame, text="Visualizations", padding=10)
       viz_frame.pack(fill="x", padx=10, pady=5)

       # Create frames for each visualization
       self.term_freq_frame = ttk.Frame(viz_frame)
       self.term_freq_frame.pack(fill="x", pady=5)
       self.term_freq_label = ttk.Label(self.term_freq_frame)
       self.term_freq_label.pack()

       self.sentiment_frame = ttk.Frame(viz_frame)
       self.sentiment_frame.pack(fill="x", pady=5)
       self.sentiment_label = ttk.Label(self.sentiment_frame)
       self.sentiment_label.pack()

       self.wordcloud_frame = ttk.Frame(viz_frame)
       self.wordcloud_frame.pack(fill="x", pady=5)
       self.wordcloud_label = ttk.Label(self.wordcloud_frame)
       self.wordcloud_label.pack()

       self.text_structure_frame1 = ttk.Frame(viz_frame)
       self.text_structure_frame1.pack(fill="x", pady=5)
       ttk.Label(self.text_structure_frame1, text="Document Structure Analysis - Part 1").pack()
       self.text_structure_label1 = ttk.Label(self.text_structure_frame1)
       self.text_structure_label1.pack()

       self.text_structure_frame2 = ttk.Frame(viz_frame)
       self.text_structure_frame2.pack(fill="x", pady=5)
       ttk.Label(self.text_structure_frame2, text="Document Structure Analysis - Part 2").pack()
       self.text_structure_label2 = ttk.Label(self.text_structure_frame2)
       self.text_structure_label2.pack()

       # Add new visualization frames
       self.readability_frame = ttk.Frame(viz_frame)
       self.readability_frame.pack(fill="x", pady=5)
       ttk.Label(self.readability_frame, text="Readability Analysis").pack()
       self.readability_label = ttk.Label(self.readability_frame)
       self.readability_label.pack()

       self.content_frame = ttk.Frame(viz_frame)
       self.content_frame.pack(fill="x", pady=5)
       ttk.Label(self.content_frame, text="Content Pattern Analysis").pack()
       self.content_label = ttk.Label(self.content_frame)
       self.content_label.pack()

       # Pack the main scrollable components
       main_canvas.pack(side="left", fill="both", expand=True)
       scrollbar.pack(side="right", fill="y")
       
       # Control Frame
       control_frame = ttk.Frame(self.root, padding=10)
       control_frame.pack(fill="x", padx=10, pady=5)
       
       self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
       self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
       
       self.analyze_button = ttk.Button(control_frame, text="Analyze Document", command=self.start_analysis)
       self.analyze_button.pack(side="right")
       
   def upload_pdf(self):
       file_path = filedialog.askopenfilename(
           filetypes=[("PDF files", "*.pdf")]
       )
       if file_path:
           self.file_label.config(text=os.path.basename(file_path))
           self.current_file = file_path
           
   def update_api_key(self):
       self.api_key = self.api_entry.get()
       self.save_api_key()
       self.llm.configure_api(self.api_key)
       messagebox.showinfo("Success", "API key saved successfully!")
           
   def process_document(self):
       self.progress.start()
       self.analyze_button.config(state="disabled")
       self.output_text.delete(1.0, tk.END)
       self.mining_text.delete(1.0, tk.END)
       
       try:
           # Extract text from PDF
           text = self.llm.extract_text_from_pdf(self.current_file)
           
           # Run LLM analysis
           llm_results = self.llm.analyze_text(
               text,
               include_key_findings=self.include_key_findings.get(),
               include_methodology=self.include_methodology.get()
           )
           
           # Run data mining analysis
           mining_results = self.data_miner.analyze_pdf(self.current_file)
           
           # Update GUI in main thread with both results
           self.root.after(0, lambda: self._update_gui_with_results(llm_results, mining_results))
           
       except Exception as e:
           self.mining_text.insert(tk.END, f"Error during analysis: {str(e)}")
       finally:
           self.progress.stop()
           self.analyze_button.config(state="normal")

   def _update_gui_with_results(self, llm_results, mining_results):
       """Update GUI with both LLM and mining results in the main thread"""
       try:
           # Display LLM analysis results
           self.output_text.delete(1.0, tk.END)
           self.output_text.insert(tk.END, llm_results)
           
           # Display data mining text results
           self.mining_text.delete(1.0, tk.END)
           self.mining_text.insert(tk.END, "=== METADATA ===\n")
           for key, value in mining_results['metadata'].items():
               self.mining_text.insert(tk.END, f"{key}: {value}\n")
               
           self.mining_text.insert(tk.END, "\n=== STATISTICS ===\n")
           for key, values in mining_results['statistics'].items():
               self.mining_text.insert(tk.END, f"{key}:\n - " + "\n - ".join(values) + "\n")
           
           self.mining_text.insert(tk.END, "\n=== CITATIONS ===\n")
           self.mining_text.insert(tk.END, "\n".join(mining_results['citations']))
           
           # Display visualizations
           self._display_plot(mining_results['term_freq_plot'], self.term_freq_label)
           self._display_plot(mining_results['sentiment_plot'], self.sentiment_label)
           self._display_plot(mining_results['wordcloud_plot'], self.wordcloud_label)
           self._display_plot(mining_results['text_structure_plot1'], self.text_structure_label1)
           self._display_plot(mining_results['text_structure_plot2'], self.text_structure_label2)
           
           # Display new visualizations
           self._display_plot(mining_results['readability_plot'], self.readability_label)
           self._display_plot(mining_results['content_plot'], self.content_label)
           
       except Exception as e:
           self.mining_text.insert(tk.END, f"\nError updating GUI: {str(e)}")

   def start_analysis(self):
       if not hasattr(self, 'current_file'):
           messagebox.showerror("Error", "Please upload a PDF file first!")
           return
           
       if not self.api_key:
           messagebox.showerror("Error", "Please configure your API key first!")
           return
           
       Thread(target=self.process_document).start()

   def _display_plot(self, plot_buffer, label_widget):
       """Helper method to display plots"""
       plot_buffer.seek(0)
       img = Image.open(plot_buffer)
       photo = ImageTk.PhotoImage(img)
       label_widget.configure(image=photo)
       label_widget.image = photo  # Keep a reference