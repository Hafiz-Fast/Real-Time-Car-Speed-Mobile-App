// RollNo: 23L-0896
// RollNo: 23L-0729

// Architecture: Calibrate with photo → Record upto 30s video → Upload to serever → Poll → Play result

import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Modal,
  TextInput,
  Alert,
  ActivityIndicator,
  Image,
  Platform,
  StatusBar as RNStatusBar,
  Dimensions,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { Video, ResizeMode } from 'expo-av';
import * as FileSystem from 'expo-file-system/legacy';

// CONFIG
const SERVER_BASE = 'http://192.168.100.25:8000';
const WS_CALIBRATE_URL = 'ws://192.168.100.25:8000/ws/calibrate';

// Maximum recording duration in seconds
const MAX_RECORD_SECONDS = 30;

// How often (ms) to poll /job/:id for progress
const POLL_INTERVAL_MS = 1500;

// Types
type AppMode =
  | 'home'           // idle — shows record/calibrate buttons
  | 'calibrating'    // calibration sub-flow
  | 'recording'      // actively recording
  | 'post-record'    // ask user zone preference before uploading
  | 'uploading'      // sending video to server
  | 'processing'     // server is working — showing progress
  | 'result';        // show annotated video

type TapPoint = { x: number; y: number };

const POINT_LABELS = [
  { label: 'TL', desc: 'Top-Left corner', color: '#FF6B6B' },
  { label: 'TR', desc: 'Top-Right corner', color: '#4ECDC4' },
  { label: 'BR', desc: 'Bottom-Right corner', color: '#FFD93D' },
  { label: 'BL', desc: 'Bottom-Left corner', color: '#6BCB77' },
];

// Calibration Screen
function CalibrationScreen({
  onCalibrated,
  onCancel,
}: {
  onCalibrated: (calJson: string) => void;
  onCancel: () => void;
}) {
  const cameraRef = useRef<CameraView | null>(null);
  const [step, setStep] = useState<'capture' | 'tap' | 'confirm'>('capture');
  const [points, setPoints] = useState<TapPoint[]>([]);
  const [capturedUri, setCapturedUri] = useState<string | null>(null);
  const [capturedB64, setCapturedB64] = useState<string | null>(null);
  const [viewLayout, setViewLayout] = useState({ width: 1, height: 1 });
  const [photoSize, setPhotoSize] = useState({ width: 1280, height: 720 });
  const [realWidth, setRealWidth] = useState('7.0');
  const [realHeight, setRealHeight] = useState('60.0');
  const [showSettings, setShowSettings] = useState(false);
  const [sending, setSending] = useState(false);
  // Delay mounting CameraView so the home-screen camera session fully releases first
  const [cameraVisible, setCameraVisible] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  useEffect(() => {
    // Give the OS ~400 ms to tear down the previous CameraView before mounting ours
    const t = setTimeout(() => setCameraVisible(true), 400);
    return () => clearTimeout(t);
  }, []);

  const captureFrame = useCallback(async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.9,
        skipProcessing: false,
      });
      if (!photo) return;
      setCapturedUri(photo.uri);
      setCapturedB64(photo.base64 ?? null);
      setPhotoSize({ width: photo.width, height: photo.height });
      setStep('tap');
    } catch {
      Alert.alert('Error', 'Could not capture frame. Try again.');
    }
  }, []);

  const handleTap = useCallback((e: any) => {
    if (step !== 'tap' || points.length >= 4) return;
    const { locationX, locationY } = e.nativeEvent;
    setPoints((prev) => {
      const next = [...prev, { x: Math.round(locationX), y: Math.round(locationY) }];
      if (next.length === 4) setStep('confirm');
      return next;
    });
  }, [step, points.length]);

  const sendCalibration = useCallback(async () => {
    if (!capturedB64 || points.length !== 4) return;
    setSending(true);

    // Scale tap coords from view space → photo pixel space
    const serverPoints = points.map((p) => [
      Math.round((p.x / viewLayout.width) * photoSize.width),
      Math.round((p.y / viewLayout.height) * photoSize.height),
    ]);

    const calConfig = {
      real_width_m: parseFloat(realWidth) || 7.0,
      real_height_m: parseFloat(realHeight) || 60.0,
      source_points: serverPoints,
      frame_width: photoSize.width,
      frame_height: photoSize.height,
    };

    // Also persist to server via WebSocket
    try {
      await new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(WS_CALIBRATE_URL);
        ws.onopen = () => ws.send(JSON.stringify({ image_base64: capturedB64, points: serverPoints, ...calConfig }));
        ws.onmessage = (ev) => {
          const payload = JSON.parse(ev.data);
          ws.close();
          if (payload.success) resolve();
          else reject(new Error(payload.error ?? 'Calibration failed'));
        };
        ws.onerror = () => reject(new Error('Connection error'));
      });
      setSending(false);
      Alert.alert('Calibrated!', `Road zone saved.\n${photoSize.width}×${photoSize.height}px`, [
        { text: 'Start Recording', onPress: () => onCalibrated(JSON.stringify(calConfig)) },
      ]);
    } catch (err: any) {
      setSending(false);
      Alert.alert('Server Error', err.message);
    }
  }, [capturedB64, points, viewLayout, photoSize, realWidth, realHeight, onCalibrated]);

  const reset = () => {
    setPoints([]);
    setCapturedUri(null);
    setCapturedB64(null);
    setCameraReady(false);  // CameraView re-mounts; wait for onCameraReady again
    setStep('capture');
  };

  const stepIdx = step === 'capture' ? 0 : step === 'tap' ? 1 : 2;
  const nextLabel = points.length < 4 ? POINT_LABELS[points.length] : null;

  return (
    <View style={cal.root}>
      <StatusBar style="light" />

      {/* Header */}
      <View style={cal.header}>
        <TouchableOpacity onPress={onCancel} hitSlop={12} style={cal.headerSide}>
          <Text style={cal.back}>✕</Text>
        </TouchableOpacity>
        <Text style={cal.title}>Calibrate Road Zone</Text>
        <TouchableOpacity onPress={() => setShowSettings(true)} hitSlop={12} style={cal.headerSide}>
          <Text style={cal.gear}>⚙</Text>
        </TouchableOpacity>
      </View>

      {/* Step indicators */}
      <View style={cal.stepRow}>
        {['Capture', 'Tap 4 Points', 'Confirm'].map((s, i) => (
          <View key={s} style={cal.stepItem}>
            <View style={[cal.stepDot, stepIdx >= i && cal.stepDotOn]}>
              <Text style={cal.stepNum}>{i + 1}</Text>
            </View>
            <Text style={[cal.stepLbl, stepIdx >= i && cal.stepLblOn]}>{s}</Text>
          </View>
        ))}
      </View>

      {step !== 'capture' && (
        <View style={{ backgroundColor: '#1a1a2e', paddingHorizontal: 12, paddingVertical: 4 }}>
          <Text style={{ color: '#4ade80', fontSize: 11, textAlign: 'center' }}>
            Photo: {photoSize.width}×{photoSize.height}px
          </Text>
        </View>
      )}

      {/* Camera / Image area */}
      <View style={{ flex: 1 }}>
        {step === 'capture' ? (
          <>
            {/* Spinner shown until camera session is ready */}
            {(!cameraVisible || !cameraReady) && (
              <View style={[StyleSheet.absoluteFill, { alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' }]}>
                <ActivityIndicator size="large" color="#1a73e8" />
                <Text style={{ color: '#888', fontSize: 13, marginTop: 12 }}>Starting camera…</Text>
              </View>
            )}
            {cameraVisible && (
              <CameraView
                style={StyleSheet.absoluteFill}
                ref={cameraRef}
                facing="back"
                onCameraReady={() => setCameraReady(true)}
              />
            )}
          </>
        ) : (
          <View
            style={StyleSheet.absoluteFill}
            onLayout={(e) => {
              const { width, height } = e.nativeEvent.layout;
              setViewLayout({ width, height });
            }}
            onStartShouldSetResponder={() => step === 'tap'}
            onResponderGrant={handleTap}
          >
            {capturedUri && (
              <Image
                source={{ uri: capturedUri }}
                style={StyleSheet.absoluteFill}
                resizeMode="cover"
              />
            )}
            <View style={cal.tint} pointerEvents="none" />

            {/* Connecting lines between tapped points */}
            {points.length >= 2 && (
              <View style={StyleSheet.absoluteFill} pointerEvents="none">
                {points.map((p, i) => {
                  if (i === 0) return null;
                  return <ConnectingLine key={i} from={points[i - 1]} to={p} />;
                })}
                {points.length === 4 && (
                  <ConnectingLine from={points[3]} to={points[0]} />
                )}
              </View>
            )}

            {/* Filled zone polygon when all 4 points set */}
            {points.length === 4 && (
              <ZonePolygon points={points} />
            )}

            {/* Tapped dots */}
            {points.map((p, i) => (
              <View
                key={i}
                pointerEvents="none"
                style={[
                  cal.dot,
                  { left: p.x - 14, top: p.y - 14, backgroundColor: POINT_LABELS[i].color },
                ]}
              >
                <Text style={cal.dotText}>{POINT_LABELS[i].label}</Text>
              </View>
            ))}

            {step === 'tap' && nextLabel && (
              <View style={cal.badge} pointerEvents="none">
                <View style={[cal.badgeDot, { backgroundColor: nextLabel.color }]} />
                <Text style={cal.badgeText}>Tap {nextLabel.label} — {nextLabel.desc}</Text>
              </View>
            )}

            {step === 'confirm' && (
              <View style={cal.confirmBadge} pointerEvents="none">
                <Text style={cal.confirmBadgeText}>✓ Zone correct? Confirm below.</Text>
              </View>
            )}
          </View>
        )}
      </View>

      {/* Actions */}
      <View style={cal.actions}>
        {step === 'capture' && (
          <TouchableOpacity
            style={[cal.btnPrimary, !cameraReady && { opacity: 0.45 }]}
            onPress={captureFrame}
            disabled={!cameraReady}
          >
            {cameraReady
              ? <Text style={cal.btnPrimaryTxt}>📸  Freeze Frame</Text>
              : <ActivityIndicator color="#fff" />}
          </TouchableOpacity>
        )}
        {step === 'tap' && (
          <View style={cal.rowBetween}>
            <Text style={cal.tapCount}>{points.length} / 4 tapped</Text>
            <TouchableOpacity style={cal.btnSecondary} onPress={reset}>
              <Text style={cal.btnSecondaryTxt}>↺  Retake</Text>
            </TouchableOpacity>
          </View>
        )}
        {step === 'confirm' && (
          <View style={cal.rowFull}>
            <TouchableOpacity style={[cal.btnSecondary, { paddingHorizontal: 20 }]} onPress={reset}>
              <Text style={cal.btnSecondaryTxt}>↺ Redo</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[cal.btnPrimary, { flex: 1, marginLeft: 12 }]}
              onPress={sendCalibration}
              disabled={sending}
            >
              {sending
                ? <ActivityIndicator color="#fff" />
                : <Text style={cal.btnPrimaryTxt}>✓  Save Calibration</Text>}
            </TouchableOpacity>
          </View>
        )}
      </View>

      {/* Settings modal */}
      <Modal visible={showSettings} transparent animationType="slide">
        <View style={cal.modalOverlay}>
          <View style={cal.modalCard}>
            <Text style={cal.modalTitle}>Road Dimensions</Text>
            <Text style={cal.modalSub}>Real-world size of the road zone.</Text>
            <Text style={cal.fieldLbl}>Road Width (m)</Text>
            <TextInput
              style={cal.input}
              value={realWidth}
              onChangeText={setRealWidth}
              keyboardType="decimal-pad"
              placeholder="7.0"
              placeholderTextColor="#555"
            />
            <Text style={cal.fieldLbl}>Zone Depth (m)</Text>
            <TextInput
              style={cal.input}
              value={realHeight}
              onChangeText={setRealHeight}
              keyboardType="decimal-pad"
              placeholder="60.0"
              placeholderTextColor="#555"
            />
            <TouchableOpacity
              style={[cal.btnPrimary, { marginTop: 8 }]}
              onPress={() => setShowSettings(false)}
            >
              <Text style={cal.btnPrimaryTxt}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// Zone polygon overlay
function ZonePolygon({ points }: { points: TapPoint[] }) {
  // Draw a translucent quadrilateral using absolute-positioned View tricks
  // A proper SVG overlay would be ideal, but RN core doesn't have it.
  // We approximate with a semi-transparent View clipped to the bounding box.
  if (points.length < 4) return null;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  return (
    <View
      pointerEvents="none"
      style={{
        position: 'absolute',
        left: minX,
        top: minY,
        width: maxX - minX,
        height: maxY - minY,
        backgroundColor: 'rgba(255, 200, 0, 0.15)',
        borderWidth: 2,
        borderColor: 'rgba(255, 200, 0, 0.7)',
      }}
    />
  );
}

// Connecting line
function ConnectingLine({ from, to }: { from: TapPoint; to: TapPoint }) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const deg = (Math.atan2(dy, dx) * 180) / Math.PI;
  return (
    <View
      style={{
        position: 'absolute',
        left: from.x,
        top: from.y - 1.5,
        width: len,
        height: 3,
        backgroundColor: 'rgba(255, 200, 0, 0.85)',
        transformOrigin: 'left center',
        transform: [{ rotate: `${deg}deg` }],
      }}
    />
  );
}

// Progress Ring
function ProgressRing({ progress }: { progress: number }) {
  return (
    <View style={ring.container}>
      <View style={ring.outer}>
        <View style={ring.inner}>
          <Text style={ring.pct}>{Math.round(progress)}%</Text>
          <Text style={ring.lbl}>Processing</Text>
        </View>
      </View>
      {/* Simple segmented indicator */}
      <View style={ring.barTrack}>
        <View style={[ring.barFill, { width: `${progress}%` as any }]} />
      </View>
    </View>
  );
}

// Home Screen
export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [micPermission, requestMicPermission] = useMicrophonePermissions();
  const cameraRef = useRef<CameraView | null>(null);
  const videoRef = useRef<any>(null);

  const [mode, setMode] = useState<AppMode>('home');
  const [calJson, setCalJson] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [resultUri, setResultUri] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showZone, setShowZone] = useState(true);
  const showZoneRef = useRef(true);           // always reflects latest showZone for closures
  const pendingJobIdRef = useRef<string | null>(null);
  const pendingVideoUriRef = useRef<string | null>(null);

  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Keeps showZoneRef in sync so uploadVideo closure always reads the latest value
  const setShowZoneSync = useCallback((val: boolean) => {
    showZoneRef.current = val;
    setShowZone(val);
  }, []);


  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const startPolling = useCallback((id: string) => {
    if (pollTimerRef.current) clearInterval(pollTimerRef.current);

    pollTimerRef.current = setInterval(async () => {
      try {
        const resp = await fetch(`${SERVER_BASE}/job/${id}`);
        const data = await resp.json();

        setJobProgress(data.progress ?? 0);

        if (data.status === 'done') {
          stopPolling();
          // Download the result to local cache
          const localPath = FileSystem.cacheDirectory + `result_${id.slice(0, 8)}.mp4`;
          const dlRes = await FileSystem.downloadAsync(
            `${SERVER_BASE}/job/${id}/download`,
            localPath,
          );
          if (dlRes.status === 200) {
            setResultUri(dlRes.uri);
            setMode('result');
            // Tell server to clean up
            fetch(`${SERVER_BASE}/job/${id}`, { method: 'DELETE' }).catch(() => { });
          } else {
            throw new Error('Download failed');
          }
        } else if (data.status === 'error') {
          stopPolling();
          setMode('home');
          setErrorMsg(data.error ?? 'Processing failed');
          Alert.alert('Processing Error', data.error ?? 'Server error');
        }
      } catch (err: any) {
        // Network hiccup — keep polling
        console.warn('Poll error:', err);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPolling]);

  const uploadVideo = useCallback(async (videoUri: string) => {
    setMode('uploading');
    setUploadProgress(0);
    setErrorMsg(null);

    try {
      const calPayload = calJson ?? JSON.stringify({
        real_width_m: 7.0, real_height_m: 60.0,
        source_points: [[0, 0], [1280, 0], [1280, 720], [0, 720]],
        frame_width: 1280, frame_height: 720,
      });

      // Use expo-file-system for multipart upload with progress tracking
      const uploadResult = await FileSystem.uploadAsync(
        `${SERVER_BASE}/process-video`,
        videoUri,
        {
          httpMethod: 'POST',
          uploadType: FileSystem.FileSystemUploadType.MULTIPART,
          fieldName: 'video',
          parameters: {
            calibration_json: calPayload,
            show_zone: showZoneRef.current ? 'true' : 'false',
          },
          mimeType: 'video/mp4',
        },
      );

      if (uploadResult.status !== 200) {
        throw new Error(`Upload failed: HTTP ${uploadResult.status}`);
      }

      const body = JSON.parse(uploadResult.body);
      if (!body.job_id) throw new Error('No job_id returned from server');

      setJobId(body.job_id);
      setMode('processing');
      setJobProgress(0);
      startPolling(body.job_id);

    } catch (err: any) {
      setMode('home');
      setErrorMsg(err?.message ?? String(err));
      Alert.alert('Upload Error', err?.message ?? String(err));
    }
  }, [calJson, startPolling]);

  const stopRecording = useCallback(() => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    cameraRef.current?.stopRecording();
    setIsRecording(false);
  }, []);

  const startRecording = useCallback(async () => {
    if (!cameraRef.current) return;
    
    // Switch to recording mode, but because we share the same CameraView tree
    // it won't unmount! 
    setMode('recording');
    setRecordSeconds(0);
    setIsRecording(true);

    // Countdown timer
    recordingTimerRef.current = setInterval(() => {
      setRecordSeconds((s) => {
        if (s + 1 >= MAX_RECORD_SECONDS) {
          stopRecording();
          return MAX_RECORD_SECONDS;
        }
        return s + 1;
      });
    }, 1000);

    try {
      // recordAsync resolves when stopRecording() is called or maxDuration reached
      const video = await cameraRef.current.recordAsync({
        maxDuration: MAX_RECORD_SECONDS,
      });
      if (video?.uri) {
        pendingVideoUriRef.current = video.uri;
        setShowZoneSync(true); // reset to default each time
        setMode('post-record');
      }
    } catch (err: any) {
      setIsRecording(false);
      setMode('home');
      if (err?.message !== 'Recording was stopped before any data could be produced') {
        Alert.alert('Recording Error', String(err?.message ?? err));
      }
    }
  }, [uploadVideo, stopRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      stopPolling();
    };
  }, []);

  // Permission gate
  if (!permission || !micPermission) return <View />;

  const needsCamera = !permission.granted;
  const needsMic = !micPermission.granted;

  if (needsCamera || needsMic) {
    return (
      <View style={S.center}>
        <StatusBar style="light" />
        <Text style={S.text}>
          {needsCamera && needsMic
            ? 'Camera & microphone permissions are required.'
            : needsCamera
            ? 'Camera permission is required.'
            : 'Microphone permission is required for video recording.'}
        </Text>
        <Text
          style={S.link}
          onPress={() => {
            if (needsCamera) requestPermission();
            else if (needsMic) requestMicPermission();
          }}
        >
          Tap to grant
        </Text>
      </View>
    );
  }

  // Calibration sub-flow
  if (mode === 'calibrating') {
    return (
      <CalibrationScreen
        onCalibrated={(json) => { setCalJson(json); setMode('home'); }}
        onCancel={() => setMode('home')}
      />
    );
  }

  // Result viewer
  if (mode === 'result' && resultUri) {
    return (
      <View style={S.root}>
        <StatusBar style="light" />
        <View style={S.topBar}>
          <Text style={S.topTitle}>🎬 Processed Result</Text>
          <TouchableOpacity
            style={S.calBtn}
            onPress={() => { setResultUri(null); setJobId(null); setMode('home'); }}
          >
            <Text style={S.calBtnTxt}>✕ Close</Text>
          </TouchableOpacity>
        </View>
        <Video
          ref={videoRef}
          source={{ uri: resultUri }}
          style={S.videoPlayer}
          useNativeControls
          resizeMode={ResizeMode.CONTAIN}
          shouldPlay
          isLooping
        />
        <View style={S.resultActions}>
          <Text style={S.resultHint}>
            Speed detection complete. Bounding boxes, zone, and speed labels are burned into the video.
          </Text>
          <TouchableOpacity
            style={[S.calBtn, { alignSelf: 'center', marginTop: 12, paddingHorizontal: 20, paddingVertical: 10 }]}
            onPress={() => { setResultUri(null); setJobId(null); setMode('home'); }}
          >
            <Text style={S.calBtnTxt}>🔁 Record Another</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Post-record: zone preference
  if (mode === 'post-record') {
    return (
      <View style={S.root}>
        <StatusBar style="light" />
        <View style={S.center}>
          <Text style={{ color: '#fff', fontSize: 22, fontWeight: '800', marginBottom: 8 }}>
            🎥 Video Ready
          </Text>
          <Text style={{ color: '#888', fontSize: 14, marginBottom: 36, textAlign: 'center', paddingHorizontal: 32 }}>
            Show the calibrated road zone overlay in the processed video?
          </Text>

          {/* Live preview illustration */}
          <View style={{
            width: 240, height: 135, borderRadius: 12, backgroundColor: '#111',
            borderWidth: 1, borderColor: '#333', marginBottom: 32,
            alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
          }}>
            <View style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '100%', backgroundColor: '#1a1a1a' }} />
            {showZone && (
              <>
                <View style={{
                  position: 'absolute',
                  bottom: 10, left: 40, right: 40, top: 30,
                  backgroundColor: 'rgba(255,180,0,0.15)',
                  borderWidth: 2, borderColor: 'rgba(255,200,0,0.7)',
                }} />
                {[
                  { top: 32, left: 40, color: '#FF6B6B', label: 'TL' },
                  { top: 32, right: 40, color: '#4ECDC4', label: 'TR' },
                  { bottom: 12, right: 40, color: '#FFD93D', label: 'BR' },
                  { bottom: 12, left: 40, color: '#6BCB77', label: 'BL' },
                ].map(({ label, color, ...pos }) => (
                  <View key={label} style={[{
                    position: 'absolute', width: 18, height: 18, borderRadius: 9,
                    backgroundColor: color, borderWidth: 2, borderColor: '#fff',
                    alignItems: 'center', justifyContent: 'center',
                  }, pos as any]}>
                    <Text style={{ color: '#fff', fontSize: 7, fontWeight: '800' }}>{label}</Text>
                  </View>
                ))}
              </>
            )}
            {/* Simulated vehicle detection box */}
            <View style={{ width: 40, height: 28, borderWidth: 2, borderColor: '#00ff88', borderRadius: 4 }}>
              <View style={{ position: 'absolute', top: -14, left: 0, backgroundColor: '#00ff88', paddingHorizontal: 3, borderRadius: 2 }}>
                <Text style={{ color: '#000', fontSize: 7, fontWeight: '700' }}>ID:3  72 km/h</Text>
              </View>
            </View>
            <Text style={{ position: 'absolute', bottom: 6, color: '#555', fontSize: 9 }}>
              {showZone ? '▪ Zone overlay ON' : '▪ Detections only'}
            </Text>
          </View>

          {/* Toggle */}
          <View style={{ flexDirection: 'row', gap: 14, marginBottom: 36, paddingHorizontal: 24, width: '100%' }}>
            <TouchableOpacity
              style={{
                flex: 1, paddingVertical: 16, borderRadius: 12, alignItems: 'center',
                backgroundColor: showZone ? '#1a73e8' : '#1e1e1e',
                borderWidth: 2, borderColor: showZone ? '#1a73e8' : '#333',
              }}
              onPress={() => setShowZoneSync(true)}
            >
              <Text style={{ fontSize: 20, marginBottom: 4 }}>🟦</Text>
              <Text style={{ color: showZone ? '#fff' : '#555', fontWeight: '700', fontSize: 14 }}>Show Zone</Text>
              <Text style={{ color: showZone ? '#a0c4ff' : '#444', fontSize: 11, marginTop: 2 }}>With overlay</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={{
                flex: 1, paddingVertical: 16, borderRadius: 12, alignItems: 'center',
                backgroundColor: !showZone ? '#1a73e8' : '#1e1e1e',
                borderWidth: 2, borderColor: !showZone ? '#1a73e8' : '#333',
              }}
              onPress={() => setShowZoneSync(false)}
            >
              <Text style={{ fontSize: 20, marginBottom: 4 }}>⬛</Text>
              <Text style={{ color: !showZone ? '#fff' : '#555', fontWeight: '700', fontSize: 14 }}>Hide Zone</Text>
              <Text style={{ color: !showZone ? '#a0c4ff' : '#444', fontSize: 11, marginTop: 2 }}>Detections only</Text>
            </TouchableOpacity>
          </View>

          {/* Confirm */}
          <TouchableOpacity
            style={{ backgroundColor: '#1a73e8', paddingVertical: 16, paddingHorizontal: 48, borderRadius: 14, marginBottom: 14 }}
            onPress={() => {
              const uri = pendingVideoUriRef.current;
              if (uri) uploadVideo(uri);
            }}
          >
            <Text style={{ color: '#fff', fontSize: 16, fontWeight: '700' }}>☁  Upload & Process</Text>
          </TouchableOpacity>

          <TouchableOpacity onPress={() => { pendingVideoUriRef.current = null; setMode('home'); }}>
            <Text style={{ color: '#555', fontSize: 13 }}>✕  Discard recording</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Processing screen
  if (mode === 'processing') {
    return (
      <View style={S.root}>
        <StatusBar style="light" />
        <View style={S.center}>
          <Text style={{ color: '#fff', fontSize: 20, fontWeight: '700', marginBottom: 24 }}>
            🔍 Analysing Video
          </Text>
          <ProgressRing progress={jobProgress} />
          <Text style={{ color: '#888', fontSize: 13, marginTop: 24, textAlign: 'center', paddingHorizontal: 32 }}>
            Running YOLO detection + DeepSort tracking on every frame.{'\n'}This may take a minute…
          </Text>
        </View>
      </View>
    );
  }

  // Uploading screen
  if (mode === 'uploading') {
    return (
      <View style={S.root}>
        <StatusBar style="light" />
        <View style={S.center}>
          <ActivityIndicator size="large" color="#1a73e8" />
          <Text style={{ color: '#fff', fontSize: 16, marginTop: 20, fontWeight: '600' }}>
            📤 Uploading video…
          </Text>
          <Text style={{ color: '#888', fontSize: 13, marginTop: 8 }}>
            Sending to {SERVER_BASE}
          </Text>
        </View>
      </View>
    );
  }

  // Home & Recording
  // These are combined to avoid unmounting the CameraView, which causes failures
  const hasCalibration = calJson !== null;
  const isRecordingMode = mode === 'recording';
  const timeLeft = MAX_RECORD_SECONDS - recordSeconds;
  const barWidth = `${(recordSeconds / MAX_RECORD_SECONDS) * 100}%` as any;

  return (
    <View style={S.root}>
      <StatusBar style="light" />

      {/* Top bar (only in home mode) */}
      {!isRecordingMode && (
        <View style={S.topBar}>
          <Text style={S.topTitle}>🚗 Speed Detector</Text>
          <TouchableOpacity style={S.calBtn} onPress={() => setMode('calibrating')}>
            <Text style={S.calBtnTxt}>⊞ Calibrate</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Camera preview */}
      <View style={S.preview}>
        <CameraView style={StyleSheet.absoluteFill} ref={cameraRef} facing="back" mode="video" mute />

        {/* Calibration status badge (only in home mode) */}
        {!isRecordingMode && (
          <View style={[S.calBadge, { backgroundColor: hasCalibration ? 'rgba(0,180,80,0.85)' : 'rgba(180,60,0,0.85)' }]}>
            <Text style={S.calBadgeTxt}>
              {hasCalibration ? '✓ Zone calibrated' : '⚠ Not calibrated — tap Calibrate'}
            </Text>
          </View>
        )}

        {/* Recording overlays */}
        {isRecordingMode && (
          <>
            {/* Timer HUD */}
            <View style={S.recHud}>
              <View style={S.recRow}>
                <View style={S.recDot} />
                <Text style={S.recTime}>
                  {String(Math.floor(recordSeconds / 60)).padStart(2, '0')}:
                  {String(recordSeconds % 60).padStart(2, '0')}
                </Text>
                <Text style={S.recLeft}>  {timeLeft}s left</Text>
              </View>
              {/* Progress bar */}
              <View style={S.recBarTrack}>
                <View style={[S.recBarFill, { width: barWidth }]} />
              </View>
            </View>

            {/* Stop button */}
            <View style={S.recControls}>
              <TouchableOpacity style={S.stopBtn} onPress={stopRecording}>
                <View style={S.stopInner} />
              </TouchableOpacity>
              <Text style={S.stopHint}>Tap to stop & process</Text>
            </View>
          </>
        )}
      </View>

      {/* Bottom panel (only in home mode) */}
      {!isRecordingMode && (
        <View style={S.bottomPanel}>
          {/* How it works */}
          <View style={S.infoRow}>
            {[
              { icon: '⊞', label: 'Calibrate\nroad zone' },
              { icon: '●', label: 'Record\n≤30 sec' },
              { icon: '☁', label: 'Auto-detect\n& measure' },
              { icon: '▶', label: 'Watch\nresult' },
            ].map(({ icon, label }, i) => (
              <View key={i} style={S.infoStep}>
                <Text style={S.infoIcon}>{icon}</Text>
                <Text style={S.infoLabel}>{label}</Text>
              </View>
            ))}
          </View>

          {errorMsg && (
            <View style={S.errorBox}>
              <Text style={S.errorTxt}>⚠ {errorMsg}</Text>
            </View>
          )}

          {/* Record button */}
          <TouchableOpacity
            style={[S.recordBtn, !hasCalibration && S.recordBtnDisabled]}
            onPress={startRecording}
            disabled={!hasCalibration}
          >
            <View style={S.recordBtnInner}>
              <View style={S.recordDot} />
            </View>
            <Text style={S.recordBtnLabel}>
              {hasCalibration ? 'Record Video' : 'Calibrate first'}
            </Text>
          </TouchableOpacity>

          <Text style={S.footNote}>
            Video is processed on your local server at {SERVER_BASE}
          </Text>
        </View>
      )}
    </View>
  );
}

// Styles
const SB_H = Platform.OS === 'android' ? (RNStatusBar.currentHeight ?? 24) : 44;
const { width: SCREEN_W } = Dimensions.get('window');

const S = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  text: { color: '#fff', marginBottom: 8 },
  link: { color: '#4ea1ff' },

  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: SB_H + 6,
    paddingBottom: 10,
    paddingHorizontal: 16,
    backgroundColor: '#111',
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e1e',
  },
  topTitle: { color: '#fff', fontSize: 17, fontWeight: '700' },
  calBtn: { backgroundColor: '#1a73e8', paddingHorizontal: 14, paddingVertical: 7, borderRadius: 8 },
  calBtnTxt: { color: '#fff', fontSize: 13, fontWeight: '600' },

  preview: { flex: 1, position: 'relative' },

  calBadge: {
    position: 'absolute',
    top: 16,
    alignSelf: 'center',
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
  },
  calBadgeTxt: { color: '#fff', fontSize: 13, fontWeight: '600' },

  bottomPanel: {
    backgroundColor: '#0d0d0d',
    paddingHorizontal: 20,
    paddingBottom: Platform.OS === 'ios' ? 34 : 20,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#1e1e1e',
  },

  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  infoStep: { alignItems: 'center', flex: 1 },
  infoIcon: { fontSize: 22, marginBottom: 4 },
  infoLabel: { color: '#777', fontSize: 10, textAlign: 'center', lineHeight: 14 },

  errorBox: {
    backgroundColor: 'rgba(220,50,50,0.15)',
    borderRadius: 8,
    padding: 10,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: 'rgba(220,50,50,0.4)',
  },
  errorTxt: { color: '#ff6b6b', fontSize: 12 },

  recordBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#cc2222',
    borderRadius: 14,
    paddingVertical: 16,
    gap: 12,
    marginBottom: 12,
  },
  recordBtnDisabled: { backgroundColor: '#444' },
  recordBtnInner: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 3,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#fff',
  },
  recordBtnLabel: { color: '#fff', fontSize: 16, fontWeight: '700' },

  footNote: { color: '#444', fontSize: 11, textAlign: 'center' },

  // ── Recording screen ──
  recHud: {
    position: 'absolute',
    top: SB_H + 8,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.72)',
    borderRadius: 12,
    padding: 12,
  },
  recRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  recDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#ff3333', marginRight: 8 },
  recTime: { color: '#fff', fontSize: 22, fontWeight: '700', fontVariant: ['tabular-nums'] },
  recLeft: { color: '#aaa', fontSize: 14 },
  recBarTrack: { height: 4, backgroundColor: '#333', borderRadius: 2, overflow: 'hidden' },
  recBarFill: { height: 4, backgroundColor: '#ff3333', borderRadius: 2 },

  recControls: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 50 : 30,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  stopBtn: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  stopInner: {
    width: 30,
    height: 30,
    borderRadius: 6,
    backgroundColor: '#fff',
  },
  stopHint: { color: '#ccc', fontSize: 13, marginTop: 10 },

  // Result
  videoPlayer: {
    flex: 1,
    backgroundColor: '#000',
  },
  resultActions: {
    padding: 20,
    backgroundColor: '#0d0d0d',
    borderTopWidth: 1,
    borderTopColor: '#1e1e1e',
  },
  resultHint: {
    color: '#aaa',
    fontSize: 13,
    textAlign: 'center',
    lineHeight: 19,
  },
});

// Progress ring styles
const ring = StyleSheet.create({
  container: { alignItems: 'center', width: SCREEN_W * 0.7 },
  outer: {
    width: 160,
    height: 160,
    borderRadius: 80,
    borderWidth: 8,
    borderColor: '#1a73e8',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  inner: { alignItems: 'center' },
  pct: { color: '#fff', fontSize: 36, fontWeight: '800' },
  lbl: { color: '#888', fontSize: 13, marginTop: 2 },
  barTrack: {
    width: '100%',
    height: 8,
    backgroundColor: '#222',
    borderRadius: 4,
    overflow: 'hidden',
  },
  barFill: { height: 8, backgroundColor: '#1a73e8', borderRadius: 4 },
});

// Calibration styles
const cal = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0a0a' },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: SB_H + 8,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: '#111',
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  headerSide: { width: 36 },
  back: { color: '#aaa', fontSize: 18 },
  title: { flex: 1, color: '#fff', fontSize: 16, fontWeight: '700', textAlign: 'center' },
  gear: { color: '#1a73e8', fontSize: 20, textAlign: 'right' },

  stepRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 10,
    gap: 28,
    backgroundColor: '#111',
  },
  stepItem: { alignItems: 'center', gap: 4 },
  stepDot: { width: 26, height: 26, borderRadius: 13, backgroundColor: '#2a2a2a', alignItems: 'center', justifyContent: 'center' },
  stepDotOn: { backgroundColor: '#1a73e8' },
  stepNum: { color: '#fff', fontSize: 12, fontWeight: '700' },
  stepLbl: { color: '#444', fontSize: 10 },
  stepLblOn: { color: '#aaa' },

  tint: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.18)' },
  dot: {
    position: 'absolute',
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#fff',
    elevation: 6,
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 4,
  },
  dotText: { color: '#fff', fontSize: 9, fontWeight: '800' },
  badge: {
    position: 'absolute',
    top: 14,
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.82)',
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 24,
    gap: 8,
  },
  badgeDot: { width: 10, height: 10, borderRadius: 5 },
  badgeText: { color: '#fff', fontSize: 13, fontWeight: '600' },
  confirmBadge: {
    position: 'absolute',
    bottom: 16,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,180,80,0.88)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  confirmBadgeText: { color: '#fff', fontWeight: '700', fontSize: 13 },

  actions: {
    padding: 16,
    backgroundColor: '#111',
    borderTopWidth: 1,
    borderTopColor: '#1e1e1e',
  },
  btnPrimary: { backgroundColor: '#1a73e8', paddingVertical: 14, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  btnPrimaryTxt: { color: '#fff', fontSize: 15, fontWeight: '700' },
  btnSecondary: { backgroundColor: '#222', paddingVertical: 14, paddingHorizontal: 16, borderRadius: 10, alignItems: 'center' },
  btnSecondaryTxt: { color: '#bbb', fontSize: 14 },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  rowFull: { flexDirection: 'row', alignItems: 'center' },
  tapCount: { color: '#fff', fontSize: 15, fontWeight: '600' },

  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.75)', justifyContent: 'flex-end' },
  modalCard: {
    backgroundColor: '#181818',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 24,
    gap: 12,
  },
  modalTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },
  modalSub: { color: '#777', fontSize: 13, lineHeight: 18 },
  fieldLbl: { color: '#aaa', fontSize: 13 },
  input: {
    backgroundColor: '#252525',
    color: '#fff',
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 8,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#3a3a3a',
  },
});