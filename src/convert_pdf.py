import os
import pymupdf as fitz
import glob

path = os.path.join(os.path.dirname(__file__), 'pdf')
output_path = os.path.join(os.path.dirname(__file__), 'pdf_images')
dpi = 300

print (path)

pdf_files = glob.glob(os.path.join(path, '*.pdf'))

if pdf_files:
    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]

        os.makedirs(os.path.join(output_path, pdf_name), exist_ok=True)

        output_pdf_path = os.path.join(output_path, pdf_name)

        try:
            doc = fitz.open(pdf_file)
            print(f"\nProcessing '{pdf_name}' ({len(doc)} pages)...")

            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                output_file = f"page-{i:03d}.png"
                pix.save(os.path.join(output_pdf_path, output_file))
            
            doc.close()
            print(f"Finished processing '{pdf_name}'. Images saved to '{output_pdf_path}'.")
        except Exception as e:
            print(f"Error processing '{pdf_name}': {e}")

else:
    print("No PDF files found.")

print("Done.")
