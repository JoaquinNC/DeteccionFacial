using System;
using System.IO;
using OpenCvSharp;

namespace DigitalSolutions
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string videosFolderPath = "Videos"; // Carpeta donde se encuentran los videos
            string outputFolderPath = "Frames"; // Carpeta para almacenar los frames

            // Asegurarse de que la carpeta de salida exista o crearla si no existe
            Directory.CreateDirectory(outputFolderPath);

            // Buscar todos los archivos de video en la carpeta de videos
            string[] videoFiles = Directory.GetFiles(videosFolderPath, "*.mp4");

            // Procesar cada archivo de video encontrado
            foreach (string videoFile in videoFiles)
            {
                string videoName = Path.GetFileNameWithoutExtension(videoFile);
                Console.WriteLine($"Procesando video: {videoName}");

                CaptureVideoAndSaveFrames(videoFile, outputFolderPath);
                ProcessFrames(outputFolderPath);
            }
        }

        static void CaptureVideoAndSaveFrames(string videoPath, string outputFolderPath)
        {
            using (var videoCapture = new VideoCapture(videoPath))
            {
                double fps = videoCapture.Fps;
                int frameCount = (int)(fps * 10); // Capturar 10 segundos
                int framesToSkip = 2; // Saltar algunos frames para reducir la carga de memoria

                for (int i = 0; i < frameCount; i++)
                {
                    Mat frame = new Mat();
                    videoCapture.Read(frame);

                    if (frame.Empty())
                        break;

                    // Saltar algunos frames para reducir la carga de memoria
                    if (i % framesToSkip != 0)
                        continue;

                    // Reducir la resolución del frame antes de guardarlo
                    Mat resizedFrame = new Mat();
                    Cv2.Resize(frame, resizedFrame, new OpenCvSharp.Size(640, 480)); // Ajusta el tamaño según sea necesario

                    string outputFileName = $"{Path.GetFileNameWithoutExtension(videoPath)}_frame_{i}.png";
                    string outputFilePath = Path.Combine(outputFolderPath, outputFileName);

                    Cv2.ImWrite(outputFilePath, resizedFrame);
                }
            }

        }

        static void ProcessFrames(string outputFolderPath)
        {
            string haarcascadeFileName = "haarcascade_frontalface_alt2.xml";
            string haarcascadePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, haarcascadeFileName);

            // Asegurarse de que la carpeta de salida exista o crearla si no existe
            Directory.CreateDirectory(outputFolderPath);

            using (var faceCascade = new CascadeClassifier(haarcascadePath))
            {
                string[] frameFiles = Directory.GetFiles(outputFolderPath);

                foreach (string frameFile in frameFiles)
                {
                    Mat frame = Cv2.ImRead(frameFile);

                    // Convertir a escala de grises para la detección facial
                    Mat gray = new Mat();
                    Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

                    // Detección frontal
                    Rect[] faces = faceCascade.DetectMultiScale(
                        gray,
                        scaleFactor: 1.1,
                        minNeighbors: 3,
                        minSize: new Size(30, 30)
                    );

                    // Verificar si se detectaron caras antes de guardar la imagen
                    if (faces.Length > 0)
                    {
                        // Dibujar rectángulos alrededor de las caras detectadas
                        foreach (Rect face in faces)
                        {
                            Cv2.Rectangle(frame, face, Scalar.Red, 2);
                        }

                        // Guardar la imagen solo si se detectan caras
                        string outputFileName = Path.GetFileName(frameFile);
                        string outputFilePath = Path.Combine(outputFolderPath, "deteccion_" + outputFileName);
                        Cv2.ImWrite(outputFilePath, frame);
                    }
                    else
                    {
                        // Si no se detectan caras, elimina la imagen "deteccion_" correspondiente
                        string outputFileName = Path.GetFileName(frameFile);
                        string outputFilePath = Path.Combine(outputFolderPath, "deteccion_" + outputFileName);
                        File.Delete(outputFilePath);
                    }
                }
            }
        }
    }
}

