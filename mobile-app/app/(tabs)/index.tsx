// app/(tabs)/index.tsx

import { CameraView, useCameraPermissions } from 'expo-camera';
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
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

const WS_URL            = 'ws://192.168.100.45:8000/ws';
const WS_CALIBRATE_URL  = 'ws://192.168.100.45:8000/ws/calibrate';
const TARGET_FPS        = 15;
const FRAME_INTERVAL_MS = Math.round(1000 / TARGET_FPS);
const GHOST_FRAMES      = 8; // frames to keep showing a vehicle after it disappears

// ─── Types ────────────────────────────────────────────────────────────────────

type Detection = {
  id: number;
  bbox: [number, number, number, number];
  speed_kmh: number;
  label: string;
  color: [number, number, number];
  lastSeen?: number;
};

type TapPoint = { x: number; y: number };
type AppMode  = 'detecting' | 'calibrating';

// ─── Calibration Screen ───────────────────────────────────────────────────────

const POINT_LABELS = [
  { label: 'TL', desc: 'Top-Left road corner',     color: '#FF6B6B' },
  { label: 'TR', desc: 'Top-Right road corner',    color: '#4ECDC4' },
  { label: 'BR', desc: 'Bottom-Right road corner', color: '#FFD93D' },
  { label: 'BL', desc: 'Bottom-Left road corner',  color: '#6BCB77' },
];

function CalibrationScreen({
  onCalibrated,
  onCancel,
}: {
  onCalibrated: () => void;
  onCancel: () => void;
}) {
  const cameraRef = useRef<CameraView | null>(null);

  const [step,          setStep]          = useState<'capture' | 'tap' | 'confirm'>('capture');
  const [points,        setPoints]        = useState<TapPoint[]>([]);
  const [capturedUri,   setCapturedUri]   = useState<string | null>(null);
  const [capturedB64,   setCapturedB64]   = useState<string | null>(null);
  const [viewLayout,    setViewLayout]    = useState({ width: 1, height: 1 });
  const [realWidth,     setRealWidth]     = useState('7.0');
  const [realHeight,    setRealHeight]    = useState('60.0');
  const [showSettings,  setShowSettings]  = useState(false);
  const [sending,       setSending]       = useState(false);

  /* Step 1 – freeze a frame */
  const captureFrame = useCallback(async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,
        skipProcessing: false,
      });
      if (!photo) return;
      setCapturedUri(photo.uri);
      setCapturedB64(photo.base64 ?? null);
      setStep('tap');
    } catch {
      Alert.alert('Error', 'Could not capture frame. Try again.');
    }
  }, []);

  /* Step 2 – tap 4 corners */
  const handleTap = useCallback(
    (e: any) => {
      if (step !== 'tap' || points.length >= 4) return;
      const { locationX, locationY } = e.nativeEvent;
      setPoints((prev) => {
        const next = [...prev, { x: Math.round(locationX), y: Math.round(locationY) }];
        if (next.length === 4) setStep('confirm');
        return next;
      });
    },
    [step, points.length],
  );

  /* Step 3 – send to server */
  const sendCalibration = useCallback(async () => {
    if (!capturedB64 || points.length !== 4) return;
    setSending(true);

    // Scale view-space taps → 1280×720 server space
    const serverPoints = points.map((p) => [
      Math.round((p.x / viewLayout.width)  * 1280),
      Math.round((p.y / viewLayout.height) * 720),
    ]);

    try {
      const ws = new WebSocket(WS_CALIBRATE_URL);
      ws.onopen = () =>
        ws.send(
          JSON.stringify({
            image_base64:  capturedB64,
            points:        serverPoints,
            real_width_m:  parseFloat(realWidth)  || 7.0,
            real_height_m: parseFloat(realHeight) || 60.0,
          }),
        );
      ws.onmessage = (ev) => {
        const payload = JSON.parse(ev.data);
        ws.close();
        setSending(false);
        if (payload.success) {
          Alert.alert(
            '✅ Calibrated!',
            `Zone saved.\nWidth: ${payload.real_width_m} m  Depth: ${payload.real_height_m} m`,
            [{ text: 'Start Detecting', onPress: onCalibrated }],
          );
        } else {
          Alert.alert('Server Error', payload.error ?? 'Calibration failed');
        }
      };
      ws.onerror = () => {
        setSending(false);
        Alert.alert('Connection Error', 'Could not reach:\n' + WS_CALIBRATE_URL);
      };
    } catch (e) {
      setSending(false);
      Alert.alert('Error', String(e));
    }
  }, [capturedB64, points, viewLayout, realWidth, realHeight, onCalibrated]);

  const reset = () => {
    setPoints([]);
    setCapturedUri(null);
    setCapturedB64(null);
    setStep('capture');
  };

  const stepIdx   = step === 'capture' ? 0 : step === 'tap' ? 1 : 2;
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

      {/* Step pills */}
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

      {/* Main area */}
      <View style={{ flex: 1 }}>
        {step === 'capture' ? (
          /* Live camera */
          <CameraView style={StyleSheet.absoluteFill} ref={cameraRef} facing="back" />
        ) : (
          /* Frozen photo + tap overlay */
          <View
            style={StyleSheet.absoluteFill}
            onLayout={(e) => {
              const { width, height } = e.nativeEvent.layout;
              setViewLayout({ width, height });
            }}
            onStartShouldSetResponder={() => step === 'tap'}
            onResponderGrant={handleTap}
          >
            {/* Captured photo as background */}
            {capturedUri && (
              <Image
                source={{ uri: capturedUri }}
                style={StyleSheet.absoluteFill}
                resizeMode="cover"
              />
            )}

            {/* Light darkening tint so dots pop */}
            <View style={cal.tint} pointerEvents="none" />

            {/* Connecting lines between points */}
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

            {/* Instruction badge */}
            {step === 'tap' && nextLabel && (
              <View style={cal.badge} pointerEvents="none">
                <View style={[cal.badgeDot, { backgroundColor: nextLabel.color }]} />
                <Text style={cal.badgeText}>
                  Tap {nextLabel.label} — {nextLabel.desc}
                </Text>
              </View>
            )}

            {step === 'confirm' && (
              <View style={cal.confirmBadge} pointerEvents="none">
                <Text style={cal.confirmBadgeText}>✓ Zone looks right? Confirm below.</Text>
              </View>
            )}
          </View>
        )}
      </View>

      {/* Action bar */}
      <View style={cal.actions}>
        {step === 'capture' && (
          <TouchableOpacity style={cal.btnPrimary} onPress={captureFrame}>
            <Text style={cal.btnPrimaryTxt}>📸  Freeze Frame</Text>
          </TouchableOpacity>
        )}
        {step === 'tap' && (
          <View style={cal.rowBetween}>
            <Text style={cal.tapCount}>{points.length} / 4 tapped</Text>
            <TouchableOpacity style={cal.btnSecondary} onPress={reset}>
              <Text style={cal.btnSecondaryTxt}>↺  Retake Photo</Text>
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
                : <Text style={cal.btnPrimaryTxt}>✓  Save &amp; Start</Text>}
            </TouchableOpacity>
          </View>
        )}
      </View>

      {/* Dimensions modal */}
      <Modal visible={showSettings} transparent animationType="slide">
        <View style={cal.modalOverlay}>
          <View style={cal.modalCard}>
            <Text style={cal.modalTitle}>Road Dimensions</Text>
            <Text style={cal.modalSub}>Real-world size of the zone you will outline.</Text>
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
            <TouchableOpacity style={[cal.btnPrimary, { marginTop: 8 }]} onPress={() => setShowSettings(false)}>
              <Text style={cal.btnPrimaryTxt}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

/* Helper: draw a line between two points using rotation */
function ConnectingLine({ from, to }: { from: TapPoint; to: TapPoint }) {
  const dx  = to.x - from.x;
  const dy  = to.y - from.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const deg = (Math.atan2(dy, dx) * 180) / Math.PI;
  return (
    <View
      style={{
        position:        'absolute',
        left:             from.x,
        top:              from.y - 1,
        width:            len,
        height:           2,
        backgroundColor: 'rgba(255,255,255,0.55)',
        transformOrigin: 'left center',
        transform:       [{ rotate: `${deg}deg` }],
      }}
    />
  );
}

// ─── Main Detection Screen ────────────────────────────────────────────────────

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef  = useRef<CameraView | null>(null);
  const wsRef      = useRef<WebSocket | null>(null);
  const busyRef    = useRef(false);
  const frameRef   = useRef(0);

  // Persist detections between frames using a map keyed by track ID
  const detMapRef = useRef<Map<number, Detection>>(new Map());
  const [detections, setDetections] = useState<Detection[]>([]);

  const [frameSize, setFrameSize] = useState({ width: 1280, height: 720 });
  const [viewSize,  setViewSize]  = useState({ width: 1,    height: 1    });
  const [wsStatus,  setWsStatus]  = useState<'connecting' | 'open' | 'closed'>('connecting');
  const [mode,      setMode]      = useState<AppMode>('detecting');
  const [fps,       setFps]       = useState(0);
  const fpsRef = useRef({ count: 0, lastTime: Date.now() });

  /* WebSocket */
  useEffect(() => {
    if (!permission?.granted || mode !== 'detecting') return;
    setWsStatus('connecting');
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        if (payload.error) return;

        if (payload.frame_size) {
          setFrameSize({ width: payload.frame_size[0], height: payload.frame_size[1] });
        }

        if (!payload.skipped && Array.isArray(payload.detections)) {
          const cur = frameRef.current;

          // Refresh seen detections
          for (const d of payload.detections as Detection[]) {
            detMapRef.current.set(d.id, { ...d, lastSeen: cur });
          }

          // Evict detections that haven't been seen for GHOST_FRAMES frames
          detMapRef.current.forEach((d, id) => {
            if (cur - (d.lastSeen ?? cur) > GHOST_FRAMES) detMapRef.current.delete(id);
          });

          setDetections(Array.from(detMapRef.current.values()));

          // FPS counter
          fpsRef.current.count++;
          const now = Date.now();
          const elapsed = now - fpsRef.current.lastTime;
          if (elapsed >= 1000) {
            setFps(Math.round((fpsRef.current.count / elapsed) * 1000));
            fpsRef.current = { count: 0, lastTime: now };
          }
        }
      } catch { /* ignore parse errors */ }
    };

    ws.onopen  = () => setWsStatus('open');
    ws.onclose = () => setWsStatus('closed');
    ws.onerror = () => setWsStatus('closed');
    return () => ws.close();
  }, [permission, mode]);

  /* Frame capture loop */
  useEffect(() => {
    if (!permission?.granted || mode !== 'detecting') return;
    const tid = setInterval(async () => {
      if (!cameraRef.current || busyRef.current) return;
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      busyRef.current = true;
      frameRef.current += 1;
      try {
        const photo = await cameraRef.current.takePictureAsync({
          base64: true, quality: 0.4, skipProcessing: true,
        });
        ws.send(JSON.stringify({ frame_id: Date.now(), image_base64: photo?.base64 }));
      } catch { /* ignore */ }
      finally { busyRef.current = false; }
    }, FRAME_INTERVAL_MS);
    return () => clearInterval(tid);
  }, [permission, mode]);

  if (!permission) return <View />;

  if (!permission.granted) {
    return (
      <View style={S.center}>
        <StatusBar style="light" />
        <Text style={S.text}>Camera permission required.</Text>
        <Text style={S.link} onPress={requestPermission}>Tap to grant</Text>
      </View>
    );
  }

  if (mode === 'calibrating') {
    return (
      <CalibrationScreen
        onCalibrated={() => { detMapRef.current.clear(); setMode('detecting'); }}
        onCancel={() => setMode('detecting')}
      />
    );
  }

  const scaleX = viewSize.width  / frameSize.width;
  const scaleY = viewSize.height / frameSize.height;
  const isOpen = wsStatus === 'open';

  return (
    <View style={S.root}>
      <StatusBar style="light" />

      <View style={S.topBar}>
        <View style={S.row}>
          <View style={[S.dot, { backgroundColor: isOpen ? '#00FF88' : '#FF4444' }]} />
          <Text style={[S.wsLbl, { color: isOpen ? '#00FF88' : '#FF4444' }]}>
            {wsStatus.toUpperCase()}
          </Text>
          <Text style={S.meta}>{'  '}| {detections.length} vehicles | {fps} FPS</Text>
        </View>
        <TouchableOpacity style={S.calBtn} onPress={() => setMode('calibrating')}>
          <Text style={S.calBtnTxt}>⊞ Calibrate</Text>
        </TouchableOpacity>
      </View>

      <View
        style={S.preview}
        onLayout={(e) => {
          const { width, height } = e.nativeEvent.layout;
          setViewSize({ width, height });
        }}
      >
        <CameraView style={StyleSheet.absoluteFill} ref={cameraRef} facing="back" />

        {detections.map((det) => {
          const [x1, y1, x2, y2] = det.bbox;
          const [b, g, r] = det.color; // OpenCV BGR
          const age     = frameRef.current - (det.lastSeen ?? frameRef.current);
          const opacity = age > 0 ? Math.max(0.3, 1 - age / GHOST_FRAMES) : 1;

          return (
            <View
              key={det.id}
              style={[
                S.box,
                {
                  left:        x1 * scaleX,
                  top:         y1 * scaleY,
                  width:       Math.max(2, (x2 - x1) * scaleX),
                  height:      Math.max(2, (y2 - y1) * scaleY),
                  borderColor: `rgb(${r},${g},${b})`,
                  opacity,
                },
              ]}
            >
              <Text style={S.label}>{det.label}</Text>
            </View>
          );
        })}

        {!isOpen && (
          <View style={S.offline}>
            <Text style={S.offlineTitle}>⚠ Disconnected</Text>
            <Text style={S.offlineSub}>{WS_URL}</Text>
          </View>
        )}
      </View>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const SB_H = Platform.OS === 'android' ? (RNStatusBar.currentHeight ?? 24) : 44;

const S = StyleSheet.create({
  root:    { flex: 1, backgroundColor: '#000' },
  topBar: {
    flexDirection:     'row',
    alignItems:        'center',
    justifyContent:    'space-between',
    paddingTop:        SB_H + 6,
    paddingBottom:     8,
    paddingHorizontal: 14,
    backgroundColor:   '#111',
  },
  row:    { flexDirection: 'row', alignItems: 'center' },
  dot:    { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  wsLbl:  { fontSize: 12, fontWeight: '700' },
  meta:   { fontSize: 12, color: '#888' },
  calBtn: { backgroundColor: '#1a73e8', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 6 },
  calBtnTxt: { color: '#fff', fontSize: 13, fontWeight: '600' },
  preview: { flex: 1 },
  box: {
    position:        'absolute',
    borderWidth:      2,
    backgroundColor: 'rgba(0,0,0,0.1)',
  },
  label: {
    color:             '#fff',
    fontSize:          11,
    fontWeight:        '600',
    paddingHorizontal: 4,
    paddingVertical:   2,
    backgroundColor:   'rgba(0,0,0,0.65)',
    alignSelf:         'flex-start',
  },
  offline: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.55)',
    alignItems:      'center',
    justifyContent:  'center',
  },
  offlineTitle: { color: '#FF4444', fontSize: 18, fontWeight: '700' },
  offlineSub:   { color: '#888', fontSize: 12, marginTop: 4 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' },
  text:   { color: '#fff', marginBottom: 8 },
  link:   { color: '#4ea1ff' },
});

const cal = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0a0a' },
  header: {
    flexDirection:     'row',
    alignItems:        'center',
    paddingTop:        SB_H + 8,
    paddingBottom:     12,
    paddingHorizontal: 16,
    backgroundColor:   '#111',
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  headerSide: { width: 36 },
  back:  { color: '#aaa', fontSize: 18 },
  title: { flex: 1, color: '#fff', fontSize: 16, fontWeight: '700', textAlign: 'center' },
  gear:  { color: '#1a73e8', fontSize: 20, textAlign: 'right' },

  stepRow: {
    flexDirection:   'row',
    justifyContent:  'center',
    alignItems:      'center',
    paddingVertical: 10,
    gap:             28,
    backgroundColor: '#111',
  },
  stepItem: { alignItems: 'center', gap: 4 },
  stepDot:  { width: 26, height: 26, borderRadius: 13, backgroundColor: '#2a2a2a', alignItems: 'center', justifyContent: 'center' },
  stepDotOn: { backgroundColor: '#1a73e8' },
  stepNum:  { color: '#fff', fontSize: 12, fontWeight: '700' },
  stepLbl:  { color: '#444', fontSize: 10 },
  stepLblOn: { color: '#aaa' },

  tint: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.2)' },

  dot: {
    position:       'absolute',
    width:           28,
    height:          28,
    borderRadius:    14,
    alignItems:      'center',
    justifyContent:  'center',
    borderWidth:     2,
    borderColor:     '#fff',
    elevation:       6,
    shadowColor:     '#000',
    shadowOpacity:   0.5,
    shadowRadius:    4,
  },
  dotText: { color: '#fff', fontSize: 9, fontWeight: '800' },

  badge: {
    position:          'absolute',
    top:               14,
    alignSelf:         'center',
    flexDirection:     'row',
    alignItems:        'center',
    backgroundColor:   'rgba(0,0,0,0.82)',
    paddingHorizontal: 14,
    paddingVertical:   9,
    borderRadius:      24,
    gap:               8,
  },
  badgeDot:  { width: 10, height: 10, borderRadius: 5 },
  badgeText: { color: '#fff', fontSize: 13, fontWeight: '600' },

  confirmBadge: {
    position:          'absolute',
    bottom:            16,
    alignSelf:         'center',
    backgroundColor:   'rgba(0,180,80,0.88)',
    paddingHorizontal: 16,
    paddingVertical:   8,
    borderRadius:      20,
  },
  confirmBadgeText: { color: '#fff', fontWeight: '700', fontSize: 13 },

  actions: {
    padding:          16,
    backgroundColor:  '#111',
    borderTopWidth:    1,
    borderTopColor:   '#1e1e1e',
  },
  btnPrimary: {
    backgroundColor: '#1a73e8',
    paddingVertical: 14,
    borderRadius:    10,
    alignItems:      'center',
    justifyContent:  'center',
  },
  btnPrimaryTxt:   { color: '#fff', fontSize: 15, fontWeight: '700' },
  btnSecondary: {
    backgroundColor:   '#222',
    paddingVertical:   14,
    paddingHorizontal: 16,
    borderRadius:      10,
    alignItems:        'center',
  },
  btnSecondaryTxt: { color: '#bbb', fontSize: 14 },
  rowBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  rowFull:    { flexDirection: 'row', alignItems: 'center' },
  tapCount:   { color: '#fff', fontSize: 15, fontWeight: '600' },

  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.75)', justifyContent: 'flex-end' },
  modalCard: {
    backgroundColor:      '#181818',
    borderTopLeftRadius:  20,
    borderTopRightRadius: 20,
    padding:              24,
    gap:                  12,
  },
  modalTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },
  modalSub:   { color: '#777', fontSize: 13, lineHeight: 18 },
  fieldLbl:   { color: '#aaa', fontSize: 13 },
  input: {
    backgroundColor:   '#252525',
    color:             '#fff',
    paddingHorizontal: 14,
    paddingVertical:   12,
    borderRadius:       8,
    fontSize:          16,
    borderWidth:        1,
    borderColor:       '#3a3a3a',
  },
});