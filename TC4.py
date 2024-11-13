import cv2
import dlib
from deepface import DeepFace

# Inicialização do detector de rostos do dlib
detector = dlib.get_frontal_face_detector()

# Carregar o vídeo
video_path = "Video_tc.mp4"
cap = cv2.VideoCapture(video_path)

# Variáveis para contagem de frames e armazenamento de resultados
frame_count = 0
anomalies = 0
activity_summary = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Detecta rostos no frame atual
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Extrai coordenadas do rosto detectado
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Análise de emoção no rosto detectado
        try:
            analysis = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
            emotion = analysis['dominant_emotion']
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            activity_summary.append(emotion)
        except Exception as e:
            print("Erro na análise de emoção:", e)
        
        # Lógica básica de detecção de anomalias
        if emotion == 'surprise' or emotion == 'fear':
            anomalies += 1

    # Exibir o frame em uma janela
    cv2.imshow('Video Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()

# Geração de relatório simples
print(f"Total de frames analisados: {frame_count}")
print(f"Anomalias detectadas: {anomalies}")
print(f"Resumo das atividades: {activity_summary}")
