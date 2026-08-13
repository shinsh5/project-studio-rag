import React, { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './pipeline-flow.css';

const NODE_WIDTH = 198;

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

const STEP_POSITIONS = {
  query: { x: 0, y: 90 },
  embedding: { x: 245, y: 90 },
  roi: { x: 490, y: 90 },
  stb: { x: 735, y: 90 },
  dedup: { x: 980, y: 90 },
  bm25: { x: 1225, y: 90 },
  prompt: { x: 1470, y: 90 },
  llm: { x: 1715, y: 90 },
  ragas: { x: 1960, y: 90 },
};

const PRESETS = [
  { id: 'roi', label: 'ROI-RAG', stb: false, bm25: false },
  { id: 'stb', label: '+ Small-to-Big', stb: true, bm25: false },
  { id: 'bm25', label: '+ BM25', stb: false, bm25: true },
  { id: 'full', label: 'STB + BM25', stb: true, bm25: true },
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

const PipelineNode = memo(({ data }) => {
  const canToggle = data.id === 'stb' || data.id === 'bm25';
  const enabled = data.id === 'stb' ? data.stbEnabled : data.bm25Enabled;

  const toggle = (event) => {
    event.stopPropagation();
    data.onToggle?.(data.id, !enabled);
  };

  return (
    <div className={`pipeline-node pipeline-node--${data.status}`}>
      <Handle type="target" position={Position.Left} className="pipeline-handle" />
      <div className="pipeline-node__header">
        <span className="pipeline-node__icon">{data.icon}</span>
        <span className="pipeline-node__status">{statusLabel(data.status)}</span>
      </div>
      <div className="pipeline-node__title">{data.title}</div>
      <div className="pipeline-node__subtitle">{data.subtitle}</div>
      {data.metric && <div className="pipeline-node__metric">{data.metric}</div>}
      {canToggle && (
        <button
          type="button"
          className={`pipeline-switch nodrag ${enabled ? 'is-on' : ''}`}
          onClick={toggle}
          aria-pressed={enabled}
          aria-label={`${data.title} ${enabled ? '비활성화' : '활성화'}`}
        >
          <span>{enabled ? 'ON' : 'OFF'}</span>
          <i />
        </button>
      )}
      {data.id === 'dedup' && (
        <div className="pipeline-node__auto">STB와 자동 연동</div>
      )}
      <Handle type="source" position={Position.Right} className="pipeline-handle" />
    </div>
  );
});

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

function PipelineFlow() {
  const root = document.getElementById('pipelineFlowRoot');
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

  const applyConfiguration = useCallback((stbEnabled, bm25Enabled) => {
    window.applyPipelineConfiguration?.({ stbEnabled, bm25Enabled });
    setState((current) => ({ ...current, stbEnabled, bm25Enabled }));
  }, []);

  const onToggle = useCallback((id, enabled) => {
    applyConfiguration(
      id === 'stb' ? enabled : state.stbEnabled,
      id === 'bm25' ? enabled : state.bm25Enabled,
    );
  }, [applyConfiguration, state.stbEnabled, state.bm25Enabled]);

  const nodes = useMemo(() => PIPELINE_STEPS.map((step) => ({
    id: step.id,
    type: 'pipeline',
    position: STEP_POSITIONS[step.id],
    draggable: true,
    selectable: true,
    style: { width: NODE_WIDTH },
    data: {
      ...step,
      status: resolveStatus(step.id, state),
      metric: metricFor(step.id, state),
      stbEnabled: state.stbEnabled,
      bm25Enabled: state.bm25Enabled,
      onToggle,
    },
  })), [state, onToggle]);

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
      type: 'smoothstep',
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
      <div className="pipeline-flow-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.14, minZoom: 0.42, maxZoom: 0.9 }}
          minZoom={0.25}
          maxZoom={1.5}
          nodesConnectable={false}
          elementsSelectable
          proOptions={{ hideAttribution: false }}
          colorMode="dark"
        >
          <Background color="#263451" gap={22} size={1.2} />
          <MiniMap
            pannable
            zoomable
            nodeColor={(node) => {
              const status = node.data.status;
              if (status === 'bypassed') return '#475569';
              if (status === 'running') return '#34d399';
              if (status === 'error' || status === 'unavailable') return '#ef4444';
              return '#3b82f6';
            }}
          />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      <div className="pipeline-flow-legend">
        <span><i className="legend-active" />Active</span>
        <span><i className="legend-running" />Running</span>
        <span><i className="legend-bypassed" />Bypassed</span>
        <span className="pipeline-flow-note">실행 순서는 고정되며 노드 위치만 자유롭게 조정할 수 있습니다.</span>
      </div>
    </div>
  );
}

const mount = document.getElementById('pipelineFlowRoot');
if (mount) createRoot(mount).render(<PipelineFlow />);
