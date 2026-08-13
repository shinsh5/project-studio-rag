import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Background,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './pipeline-flow.css';

const NODE_WIDTH = 150;
const NODE_HEIGHT = 172;
const NODE_GAP_X = 30;
const NODE_GAP_Y = 24;
const COLUMNS = 3;

const PIPELINE_STEPS = [
  { id: 'query', title: 'Query', subtitle: '사용자 질문', icon: 'Q' },
  { id: 'embedding', title: 'Query Embedding', subtitle: 'MiniLM · 384-D', icon: 'E' },
  { id: 'roi', title: 'ROI-RAG', subtitle: 'FAISS Top-K Evidence Units', icon: 'R', core: true },
  { id: 'stb', title: 'Small-to-Big', subtitle: 'Leaf → Parent 확장', icon: 'S', optional: true },
  { id: 'dedup', title: 'Parent Dedup', subtitle: '중복 Parent 제거', icon: 'D', dependent: true },
  { id: 'bm25', title: 'BM25', subtitle: '문장 단위 근거 압축', icon: 'B', optional: true },
  { id: 'prompt', title: 'Prompt Builder', subtitle: 'Grounded context 조립', icon: 'P' },
  { id: 'llm', title: 'LLM Generation', subtitle: '로컬 Ollama', icon: 'L' },
  { id: 'ragas', title: 'RAGAS', subtitle: 'Gemini 평가', icon: 'G' },
];

/*
 * 3열 × 3행 뱀(boustrophedon) 배치.
 * 짝수 행은 좌 → 우, 홀수 행은 우 → 좌로 흘러서 행이 바뀌는 지점의
 * 두 노드가 같은 열에 놓인다. 덕분에 모든 연결선이 수평 아니면 수직인
 * 직선으로 그려진다 (좌 → 우로만 흘리면 행 끝에서 반대편까지 되돌아오는
 * 긴 대각선이 생긴다).
 */
const STEP_LAYOUT = Object.fromEntries(PIPELINE_STEPS.map((step, index) => {
  const row = Math.floor(index / COLUMNS);
  const offset = index % COLUMNS;
  const flowsRight = row % 2 === 0;
  const column = flowsRight ? offset : COLUMNS - 1 - offset;

  return [step.id, {
    position: {
      x: column * (NODE_WIDTH + NODE_GAP_X),
      y: row * (NODE_HEIGHT + NODE_GAP_Y),
    },
    // 행 끝에서는 아래로 빠지고, 다음 행 첫 노드는 위에서 받는다.
    targetPosition: offset === 0 && row > 0
      ? Position.Top
      : (flowsRight ? Position.Left : Position.Right),
    sourcePosition: offset === COLUMNS - 1
      ? Position.Bottom
      : (flowsRight ? Position.Right : Position.Left),
  }];
}));

// + BM25는 Small-to-Big 위에 BM25를 얹은 조합(구 STB + BM25)을 그대로 맡는다.
const PRESETS = [
  { id: 'roi', label: 'ROI-RAG', stb: false, bm25: false },
  { id: 'stb', label: '+ Small-to-Big', stb: true, bm25: false },
  { id: 'bm25', label: '+ BM25', stb: true, bm25: true },
];

function statusLabel(status) {
  return {
    idle: 'READY',
    active: 'ACTIVE',
    running: 'RUNNING',
    completed: 'DONE',
    bypassed: 'BYPASS',
    unavailable: 'NO INDEX',
    error: 'ERROR',
  }[status] || status.toUpperCase();
}

// 노드는 상태 표시 전용이다. 켜고 끄는 조작은 상단 프리셋 버튼이 담당한다.
const PipelineNode = memo(({ data }) => (
  <div className={`pipeline-node pipeline-node--${data.status}`}>
    <Handle type="target" position={data.targetPosition} className="pipeline-handle" />
    <div className="pipeline-node__header">
      <span className="pipeline-node__icon">{data.icon}</span>
      <span className="pipeline-node__status">{statusLabel(data.status)}</span>
    </div>
    <div className="pipeline-node__title">{data.title}</div>
    <div className="pipeline-node__subtitle">{data.subtitle}</div>
    {data.metric && <div className="pipeline-node__metric">{data.metric}</div>}
    {data.id === 'dedup' && <div className="pipeline-node__auto">STB와 자동 연동</div>}
    <Handle type="source" position={data.sourcePosition} className="pipeline-handle" />
  </div>
));

const nodeTypes = { pipeline: PipelineNode };

function metricFor(stepId, state) {
  const metrics = state.metrics || {};
  const pipeline = metrics.pipeline || {};
  const ragas = state.ragas || {};
  switch (stepId) {
    case 'query':
      return state.query ? `${state.query.length} chars` : '질문 대기';
    case 'embedding':
      return pipeline.embedding_dimension ? `${pipeline.embedding_dimension}-D vector` : 'all-MiniLM-L6-v2';
    case 'roi':
      return pipeline.retrieved_eus != null ? `${pipeline.retrieved_eus} EUs retrieved` : 'Top-K = 3';
    case 'stb':
      return pipeline.expanded_parents != null ? `${pipeline.expanded_parents} parents expanded` : '선택 기능';
    case 'dedup':
      return pipeline.unique_parents != null
        ? `${pipeline.parent_duplicates_removed} removed · ${pipeline.unique_parents} unique`
        : 'STB 활성화 시 실행';
    case 'bm25':
      if (pipeline.bm25_sentences_before == null) return '선택 기능';
      return `${pipeline.bm25_sentences_before} → ${pipeline.bm25_sentences_after} sentences`;
    case 'prompt':
      return pipeline.prompt_chars != null ? `${pipeline.prompt_chars.toLocaleString()} chars` : '근거 프롬프트';
    case 'llm':
      if (pipeline.latency_ms == null) return state.llmModel || 'Ollama';
      return `${pipeline.latency_ms.toLocaleString()} ms${pipeline.cache_hit ? ' · cache' : ''}`;
    case 'ragas':
      if (ragas.faithfulness == null) return '평가 버튼으로 실행';
      return `F ${ragas.faithfulness.toFixed(4)}${ragas.answer_relevancy == null ? '' : ` · AR ${ragas.answer_relevancy.toFixed(4)}`}`;
    default:
      return '';
  }
}

function resolveStatus(stepId, state) {
  if (state.errorNode === stepId) return 'error';
  if (stepId === 'stb') {
    if (!state.stbEnabled) return 'bypassed';
    if (state.stbAvailable === false) return 'unavailable';
  }
  if (stepId === 'dedup' && !state.stbEnabled) return 'bypassed';
  if (stepId === 'bm25' && !state.bm25Enabled) return 'bypassed';
  if (stepId === 'ragas') {
    if (state.ragasPhase === 'running') return 'running';
    if (state.ragasPhase === 'completed') return 'completed';
    if (state.ragasPhase === 'error') return 'error';
    return 'idle';
  }
  if (state.phase === 'running') {
    const runningOrder = ['query', 'embedding', 'roi', 'stb', 'dedup', 'bm25', 'prompt', 'llm'];
    const current = runningOrder.indexOf(state.activeNode);
    const index = runningOrder.indexOf(stepId);
    if (index >= 0 && index < current) return 'completed';
    if (stepId === state.activeNode) return 'running';
    return stepId === 'roi' ? 'active' : 'idle';
  }
  if (state.phase === 'completed' && stepId !== 'ragas') {
    return stepId === 'stb' && !state.stbEnabled
      ? 'bypassed'
      : stepId === 'dedup' && !state.stbEnabled
        ? 'bypassed'
        : stepId === 'bm25' && !state.bm25Enabled
          ? 'bypassed'
          : 'completed';
  }
  return stepId === 'roi' ? 'active' : 'idle';
}

// maxZoom이 실제로 크기를 정한다. fitView는 폭에 맞춰 다시 확대하므로
// 노드 CSS만 줄이면 배율만 올라가고 화면상 크기는 그대로다. 0.85로 눌러
// 전체 폭보다 작게, 가운데 정렬로 그린다.
const FIT_VIEW_OPTIONS = { padding: 0.05, minZoom: 0.25, maxZoom: 0.85 };

function PipelineFlow() {
  const root = document.getElementById('pipelineFlowRoot');
  const canvasRef = useRef(null);
  const [flow, setFlow] = useState(null);
  const [state, setState] = useState({
    stbEnabled: false,
    bm25Enabled: false,
    stbAvailable: null,
    phase: 'idle',
    activeNode: null,
    errorNode: null,
    metrics: null,
    query: '',
    ragasPhase: 'idle',
    ragas: null,
    llmModel: root?.dataset.llmModel || 'Ollama',
  });

  const refreshStbAvailability = useCallback(async () => {
    try {
      const response = await fetch('/api/current-index?strategy=small_to_big');
      const data = await response.json();
      setState((current) => ({ ...current, stbAvailable: data.status === 'success' }));
    } catch {
      setState((current) => ({ ...current, stbAvailable: false }));
    }
  }, []);

  useEffect(() => {
    refreshStbAvailability();
    const configHandler = (event) => {
      setState((current) => ({ ...current, ...event.detail }));
      refreshStbAvailability();
    };
    const runHandler = (event) => {
      setState((current) => ({ ...current, ...event.detail }));
    };
    window.addEventListener('pipeline-config-changed', configHandler);
    window.addEventListener('pipeline-run-state', runHandler);
    window.__pipelineFlowMounted = true;
    window.dispatchEvent(new CustomEvent('pipeline-flow-ready'));
    return () => {
      window.removeEventListener('pipeline-config-changed', configHandler);
      window.removeEventListener('pipeline-run-state', runHandler);
    };
  }, [refreshStbAvailability]);

  useEffect(() => {
    if (state.phase !== 'running') return undefined;
    const order = ['embedding', 'roi'];
    if (state.stbEnabled) order.push('stb', 'dedup');
    if (state.bm25Enabled) order.push('bm25');
    order.push('prompt', 'llm');
    let index = 0;
    setState((current) => ({ ...current, activeNode: order[0] }));
    const timer = window.setInterval(() => {
      index = Math.min(index + 1, order.length - 1);
      setState((current) => current.phase === 'running'
        ? { ...current, activeNode: order[index] }
        : current);
    }, 420);
    return () => window.clearInterval(timer);
  }, [state.phase, state.stbEnabled, state.bm25Enabled]);

  // 확대/축소·패닝을 막았으므로 뷰를 되돌릴 방법이 없다.
  // 카드 폭이 변하면 다시 fit 시켜 항상 전체가 보이도록 유지한다.
  useEffect(() => {
    if (!flow || !canvasRef.current) return undefined;
    const observer = new ResizeObserver(() => flow.fitView(FIT_VIEW_OPTIONS));
    observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [flow]);

  const applyConfiguration = useCallback((stbEnabled, bm25Enabled) => {
    window.applyPipelineConfiguration?.({ stbEnabled, bm25Enabled });
    setState((current) => ({ ...current, stbEnabled, bm25Enabled }));
  }, []);

  const nodes = useMemo(() => PIPELINE_STEPS.map((step) => {
    const layout = STEP_LAYOUT[step.id];
    return {
      id: step.id,
      type: 'pipeline',
      position: layout.position,
      draggable: false,
      selectable: true,
      style: { width: NODE_WIDTH },
      data: {
        ...step,
        status: resolveStatus(step.id, state),
        metric: metricFor(step.id, state),
        targetPosition: layout.targetPosition,
        sourcePosition: layout.sourcePosition,
      },
    };
  }), [state]);

  const edges = useMemo(() => PIPELINE_STEPS.slice(0, -1).map((step, index) => {
    const target = PIPELINE_STEPS[index + 1];
    const sourceStatus = resolveStatus(step.id, state);
    const targetStatus = resolveStatus(target.id, state);
    const bypassed = sourceStatus === 'bypassed' || targetStatus === 'bypassed';
    const running = sourceStatus === 'running' || targetStatus === 'running';
    return {
      id: `${step.id}-${target.id}`,
      source: step.id,
      target: target.id,
      type: 'straight',
      animated: running,
      markerEnd: { type: MarkerType.ArrowClosed, color: bypassed ? '#475569' : '#60a5fa' },
      style: {
        stroke: bypassed ? '#475569' : running ? '#34d399' : '#60a5fa',
        strokeWidth: running ? 3 : 2,
        strokeDasharray: bypassed ? '7 6' : undefined,
      },
    };
  }), [state]);

  const activePreset = PRESETS.find(
    (preset) => preset.stb === state.stbEnabled && preset.bm25 === state.bm25Enabled,
  )?.id;

  return (
    <div className="pipeline-flow-shell">
      <div className="pipeline-flow-toolbar">
        <div>
          <div className="pipeline-flow-kicker">PIPELINE CONFIGURATION</div>
          <div className="pipeline-flow-help">전체 모듈을 표시하고 선택 기능은 bypass합니다.</div>
        </div>
        <div className="pipeline-presets" role="group" aria-label="파이프라인 프리셋">
          {PRESETS.map((preset) => (
            <button
              type="button"
              key={preset.id}
              className={activePreset === preset.id ? 'is-active' : ''}
              onClick={() => applyConfiguration(preset.stb, preset.bm25)}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
      <div className="pipeline-flow-canvas" ref={canvasRef}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onInit={setFlow}
          fitView
          fitViewOptions={FIT_VIEW_OPTIONS}
          minZoom={FIT_VIEW_OPTIONS.minZoom}
          maxZoom={FIT_VIEW_OPTIONS.maxZoom}
          /* fit된 뷰에 고정: 휠·핀치·더블클릭 확대와 드래그 패닝을 모두 끈다.
             preventScrolling=false 라야 캔버스 위에서도 페이지 스크롤이 먹는다. */
          zoomOnScroll={false}
          zoomOnPinch={false}
          zoomOnDoubleClick={false}
          panOnDrag={false}
          panOnScroll={false}
          preventScrolling={false}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          proOptions={{ hideAttribution: false }}
          colorMode="dark"
        >
          <Background color="#263451" gap={22} size={1.2} />
        </ReactFlow>
      </div>
      <div className="pipeline-flow-legend">
        <span><i className="legend-active" />Active</span>
        <span><i className="legend-running" />Running</span>
        <span><i className="legend-done" />Done</span>
        <span><i className="legend-bypassed" />Bypassed</span>
        <span className="pipeline-flow-note">실행 순서와 배치는 고정이며, 선택 기능만 스위치나 프리셋으로 켜고 끕니다.</span>
      </div>
    </div>
  );
}

const mount = document.getElementById('pipelineFlowRoot');
if (mount) createRoot(mount).render(<PipelineFlow />);
