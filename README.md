# PaperScan – Webcam Document Scanner

Turn your ordinary webcam into a document scanner.

PaperScan is a Windows desktop application built with Python, OpenCV and Tkinter. It detects paper documents through a webcam, corrects perspective, enhances image quality and exports scanned documents as PDF or images.

## Features

- Live webcam preview
- Built-in and external USB webcam support
- Automatic document edge detection
- Manual four-corner adjustment
- Perspective correction
- Color, Grayscale and Black & White scanning modes
- Adjustable B&W enhancement settings
- Multi-page document scanning
- A4 PDF export at 300 DPI
- Image export
- Scrollable application controls

## Getting Started

Install the required dependencies:

```bash
pip install opencv-python pillow numpy
```

Run the application:

```bash
python paperscan.py
```

## Build Windows EXE

Install PyInstaller:

```bash
pip install pyinstaller
```

Build:

```bash
pyinstaller --noconsole --onefile --name PaperScan paperscan.py
```

The executable will be generated inside the `dist` folder.

## Contributions & Community Feedback

**Help make PaperScan better!**

This is an open project, and contributions from developers, designers and users are welcome.

I'm looking for help with:

- Improving document detection accuracy.
- Enhancing image quality and text clarity.
- Better shadow and background removal.
- Reducing image noise without losing small text.
- Improving perspective correction.
- Supporting more webcam models and resolutions.
- Optimizing scanning performance.
- Improving the user interface and user experience.
- Adding useful scanning features.

### How You Can Help

- Report bugs and unexpected behaviour.
- Suggest new features and improvements.
- Test the application with different webcams.
- Share ideas for better image-processing techniques.
- Submit pull requests with code improvements.

When reporting an issue, please include your Windows version, webcam model, steps to reproduce the problem and sample output where possible. Remove any personal or confidential information from sample documents.

## Feedback

Have an idea that could make PaperScan better?

Open a GitHub Issue and share your suggestions.

All constructive feedback, testing and contributions are appreciated.

**Let's build a better free webcam document scanner together.**

## License

License information will be added once the project license is selected.
