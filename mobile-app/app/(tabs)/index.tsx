import { CameraView, useCameraPermissions } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';

const WS_URL = 'ws://192.168.100.45:8000/ws';
const TARGET_FPS = 15;
const FRAME_INTERVAL_MS = Math.round(1000 / TARGET_FPS);

type Detection = {
  id: number;
  bbox: [number, number, number, number];
  speed_kmh: number;
  label: string;
  color: [number, number, number];
};

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const busyRef = useRef(false);

  const [detections, setDetections] = useState<Detection[]>([]);
  const [frameSize, setFrameSize] = useState({ width: 1, height: 1 });
  const [viewSize, setViewSize] = useState({ width: 1, height: 1 });

  const [wsStatus, setWsStatus] = useState<'connecting' | 'open' | 'closed'>('connecting');

  useEffect(() => {
    if (!permission || !permission.granted) {
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.frame_size) {
          setFrameSize({
            width: payload.frame_size[0],
            height: payload.frame_size[1],
          });
        }
        if (payload.detections) {
          setDetections(payload.detections);
        }
      } catch (err) {
        console.log('WS parse error', err);
      }
    };

    ws.onopen = () => setWsStatus('open');
    ws.onclose = () => setWsStatus('closed');
    ws.onerror = () => setWsStatus('closed');


    return () => {
      ws.close();
    };
  }, [permission]);

  useEffect(() => {
    let timerId: ReturnType<typeof setInterval> | null = null;

    const startLoop = () => {
      timerId = setInterval(async () => {
        if (!cameraRef.current || busyRef.current) {
          return;
        }
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          return;
        }

        busyRef.current = true;
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.4,
            skipProcessing: true,
          });

          ws.send(
            JSON.stringify({
              frame_id: Date.now(),
              image_base64: photo.base64,
            })
          );
        } catch (err) {
          console.log('Capture error', err);
        } finally {
          busyRef.current = false;
        }
      }, FRAME_INTERVAL_MS);
    };

    if (permission && permission.granted) {
      startLoop();
    }

    return () => {
      if (timerId) {
        clearInterval(timerId);
      }
    };
  }, [permission]);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.text}>Camera permission is required.</Text>
        <Text style={styles.link} onPress={requestPermission}>
          Tap to grant permission
        </Text>
      </SafeAreaView>
    );
  }

  const scaleX = viewSize.width / frameSize.width;
  const scaleY = viewSize.height / frameSize.height;

  return (
    <SafeAreaView style={styles.container}>

      <Text style={{ color: wsStatus === 'open' ? 'lime' : 'red', padding: 8 }}>
        WS: {wsStatus} | Detections: {detections.length}
      </Text>

      <View
        style={styles.preview}
        onLayout={(event) => {
          const { width, height } = event.nativeEvent.layout;
          setViewSize({ width, height });
        }}
      >
        <CameraView style={StyleSheet.absoluteFill} ref={cameraRef} facing="back" />

        {detections.map((det) => {
          const [x1, y1, x2, y2] = det.bbox;
          const boxStyle = {
            left: x1 * scaleX,
            top: y1 * scaleY,
            width: Math.max(1, (x2 - x1) * scaleX),
            height: Math.max(1, (y2 - y1) * scaleY),
            borderColor: `rgb(${det.color[2]}, ${det.color[1]}, ${det.color[0]})`
          };

          return (
            <View key={det.id} style={[styles.box, boxStyle]}>
              <Text style={styles.label}>{det.label}</Text>
            </View>
          );
        })}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  preview: {
    flex: 1,
  },
  box: {
    position: 'absolute',
    borderWidth: 2,
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
  },
  label: {
    color: '#fff',
    fontSize: 12,
    padding: 2,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
  },
  text: {
    color: '#fff',
    marginBottom: 8,
  },
  link: {
    color: '#4ea1ff',
  },
});
