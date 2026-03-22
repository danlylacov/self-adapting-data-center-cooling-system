import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { HelpHint } from './HelpHint.jsx'
import { HOME_HINTS as HH } from './hintsRu.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
const DELTA_TIME_MIN = 1
const DELTA_TIME_MAX = 300

const clamp = (value, min, max) => Math.max(min, Math.min(max, Number(value)))

const movingAverage = (rows, key, windowSize = 5) => {
  if (!rows.length) return []
  return rows.map((row, idx) => {
    const from = Math.max(0, idx - windowSize + 1)
    const segment = rows.slice(from, idx + 1)
    const avg = segment.reduce((acc, item) => acc + (Number(item[key]) || 0), 0) / segment.length
    return avg
  })
}

const initialScenario = {
  steps: 360,
  deltaTime: 10,
  setpoint: 22,
  fanSpeed: 65,
  coolingMode: 'mixed',
  meanLoad: 0.55,
  stdLoad: 0.12,
  outsideTemp: 24,
  useDatasetLoad: false,
}

const initialRealism = {
  mode: 'realistic',
  use_dynamic_crac_power: true,
  room_temp_clip_min: 10,
  room_temp_clip_max: 42,
  chip_temp_clip_multiplier: 1.18,
}

export default function HomePage() {
  const [scenario, setScenario] = useState(initialScenario)
  const [runData, setRunData] = useState([])
  const [progress, setProgress] = useState({ step: 0, total: 0, running: false })
  const [finalState, setFinalState] = useState(null)
  const [datasets, setDatasets] = useState([])
  const [selectedLoadDataset, setSelectedLoadDataset] = useState('')
  const [loadFile, setLoadFile] = useState(null)
  const [realism, setRealism] = useState(initialRealism)
  const [error, setError] = useState('')

  const request = async (method, path, data) => {
    const response = await axios({ method, url: `${API_BASE}${path}`, data })
    return response.data
  }

  const pushRealismToApi = async (payload) => {
    await request('post', '/realism/mode', { mode: payload.mode })
    return request('post', '/realism/params', {
      use_dynamic_crac_power: payload.use_dynamic_crac_power,
      room_temp_clip_min: Number(payload.room_temp_clip_min),
      room_temp_clip_max: Number(payload.room_temp_clip_max),
      chip_temp_clip_multiplier: Number(payload.chip_temp_clip_multiplier),
    })
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const r = await request('get', '/realism')
        if (!cancelled) setRealism(r)
      } catch (e) {
        if (!cancelled) setError(String(e?.response?.data?.detail ?? e?.message ?? e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const chartData = useMemo(() => {
    if (!runData.length) return []
    const win = Math.min(21, Math.max(5, Math.floor(runData.length / 20)))
    const chipMa = movingAverage(runData, 'avgChip', win)
    const pueMa = movingAverage(runData, 'pue', win)
    const roomMa = movingAverage(runData, 'room', win)
    return runData.map((row, idx) => ({
      ...row,
      timeMin: ((row.step ?? idx) * (scenario.deltaTime ?? 1)) / 60,
      avgChipMa: chipMa[idx],
      pueMa: pueMa[idx],
      roomMa: roomMa[idx],
    }))
  }, [runData, scenario.deltaTime])

  const aggregates = useMemo(() => {
    if (!finalState) return []
    const t = finalState.telemetry ?? {}
    return [
      ['Финальный шаг', String(finalState.step ?? 0)],
      ['PUE', Number(t.pue_real ?? 0).toFixed(3)],
      ['Темп. зала', Number(finalState.room?.temperature ?? 0).toFixed(2) + ' C'],
      ['Средняя темп. чипа', Number(t.avg_chip_temp ?? 0).toFixed(2) + ' C'],
      ['Риск перегрева', (Number(t.overheat_risk ?? 0) * 100).toFixed(1) + ' %'],
    ]
  }, [finalState])

  const runFastScenario = async () => {
    setError('')
    if (!Number.isFinite(scenario.steps) || scenario.steps < 1) {
      setError('Количество шагов должно быть >= 1')
      return
    }
    if (!Number.isFinite(scenario.deltaTime) || scenario.deltaTime < DELTA_TIME_MIN || scenario.deltaTime > DELTA_TIME_MAX) {
      setError(`Delta time должен быть в диапазоне ${DELTA_TIME_MIN}-${DELTA_TIME_MAX}`)
      return
    }
    if (scenario.useDatasetLoad && !selectedLoadDataset) {
      setError('Выбран режим датасета, но файл нагрузки не выбран')
      return
    }
    setRunData([])
    setFinalState(null)
    setProgress({ step: 0, total: scenario.steps, running: true })
    try {
      const [ds, currentRealism] = await Promise.all([
        request('get', '/datasets'),
        request('get', '/realism'),
      ])
      setDatasets(ds.datasets ?? [])
      setRealism(currentRealism)
      await request('post', '/simulation/stop', {})
      await request('post', '/simulation/reset', {})
      await request('post', '/cooling/mode', { mode: scenario.coolingMode })
      await request('post', '/cooling/setpoint', { temperature: scenario.setpoint })
      await request('post', '/cooling/fanspeed', { speed: scenario.fanSpeed })
      if (scenario.useDatasetLoad && selectedLoadDataset) {
        await request('post', '/datasets/load/select', { path: selectedLoadDataset })
        await request('post', '/load/mode', { mode: 'dataset' })
      } else {
        await request('post', '/load/mode', { mode: 'random' })
        await request('post', '/load/params', { mean_load: scenario.meanLoad, std_load: scenario.stdLoad })
      }
      await request('post', '/environment/weather-mode', { mode: 'manual' })
      await request('post', '/environment/outside', { temperature: scenario.outsideTemp, humidity: 40, wind_speed: 0 })

      const points = []
      const chunk = 5
      let done = 0
      const safeDelta = clamp(scenario.deltaTime, DELTA_TIME_MIN, DELTA_TIME_MAX)
      while (done < scenario.steps) {
        const current = Math.min(chunk, scenario.steps - done)
        await request('post', '/simulation/step', { steps: current, delta_time: safeDelta })
        done += current

        const [state, telemetry] = await Promise.all([
          request('get', '/simulation/state'),
          request('get', '/simulation/telemetry'),
        ])

        points.push({
          step: state.step,
          room: state?.room?.temperature ?? 0,
          outside: state?.room?.outside_temperature ?? 0,
          pue: telemetry?.pue ?? 0,
          avgChip: state?.telemetry?.avg_chip_temp ?? 0,
          overheatRisk: (state?.telemetry?.overheat_risk ?? 0) * 100,
          coolingPower: telemetry?.cooling?.power_consumption ?? 0,
          totalPowerKw: state?.telemetry?.total_power_kw ?? 0,
        })

        setProgress({ step: done, total: scenario.steps, running: true })
        setRunData([...points])
      }

      const state = await request('get', '/simulation/state')
      setFinalState(state)
    } catch (e) {
      setError(e?.response?.data?.detail ?? String(e))
    } finally {
      setProgress((prev) => ({ ...prev, running: false }))
    }
  }

  const runSingleStep = async () => {
    setError('')
    if (!Number.isFinite(scenario.deltaTime) || scenario.deltaTime < DELTA_TIME_MIN || scenario.deltaTime > DELTA_TIME_MAX) {
      setError(`Delta time должен быть в диапазоне ${DELTA_TIME_MIN}-${DELTA_TIME_MAX}`)
      return
    }
    try {
      if (!finalState) {
        await request('post', '/simulation/stop', {})
        await request('post', '/simulation/reset', {})
        await request('post', '/cooling/mode', { mode: scenario.coolingMode })
        await request('post', '/cooling/setpoint', { temperature: scenario.setpoint })
        await request('post', '/cooling/fanspeed', { speed: scenario.fanSpeed })
        if (scenario.useDatasetLoad && selectedLoadDataset) {
          await request('post', '/datasets/load/select', { path: selectedLoadDataset })
          await request('post', '/load/mode', { mode: 'dataset' })
        } else {
          await request('post', '/load/mode', { mode: 'random' })
          await request('post', '/load/params', { mean_load: scenario.meanLoad, std_load: scenario.stdLoad })
        }
        await request('post', '/environment/weather-mode', { mode: 'manual' })
        await request('post', '/environment/outside', { temperature: scenario.outsideTemp, humidity: 40, wind_speed: 0 })
      }

      const safeDelta = clamp(scenario.deltaTime, DELTA_TIME_MIN, DELTA_TIME_MAX)
      await request('post', '/simulation/step', { steps: 1, delta_time: safeDelta })
      const [state, telemetry] = await Promise.all([
        request('get', '/simulation/state'),
        request('get', '/simulation/telemetry'),
      ])
      setFinalState(state)
      setRunData((prev) =>
        [
          ...prev,
          {
            step: state.step,
            room: state?.room?.temperature ?? 0,
            outside: state?.room?.outside_temperature ?? 0,
            pue: telemetry?.pue ?? 0,
            avgChip: state?.telemetry?.avg_chip_temp ?? 0,
            overheatRisk: (state?.telemetry?.overheat_risk ?? 0) * 100,
            coolingPower: telemetry?.cooling?.power_consumption ?? 0,
            totalPowerKw: state?.telemetry?.total_power_kw ?? 0,
          },
        ].slice(-1000),
      )
      setProgress({ step: state.step, total: scenario.steps, running: false })
    } catch (e) {
      setError(e?.response?.data?.detail ?? String(e))
    }
  }

  const uploadLoadDataset = async () => {
    setError('')
    if (!loadFile) {
      setError('Выбери CSV файл нагрузки')
      return
    }
    try {
      const form = new FormData()
      form.append('file', loadFile)
      await axios.post(`${API_BASE}/datasets/load/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const ds = await request('get', '/datasets')
      setDatasets(ds.datasets ?? [])
    } catch (e) {
      setError(e?.response?.data?.detail ?? String(e))
    }
  }

  const applyRealism = async () => {
    setError('')
    try {
      const updated = await pushRealismToApi(realism)
      setRealism(updated)
    } catch (e) {
      setError(e?.response?.data?.detail ?? String(e))
    }
  }

  const stopRun = async () => {
    try {
      await request('post', '/simulation/stop', {})
      setProgress((prev) => ({ ...prev, running: false }))
    } catch (e) {
      setError(e?.response?.data?.detail ?? String(e))
    }
  }

  const progressPct = progress.total > 0 ? Math.round((progress.step / progress.total) * 100) : 0
  const servers = finalState?.rack?.servers ?? []

  return (
    <div className="layout">
      <header className="topbar">
        <h1>Быстрый прогон цифрового двойника ЦОДа</h1>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="quickPanel">
        <h2>Сценарий прогона</h2>
        <div className="controlGrid">
          <label>
            <span className="labelRow">
              Количество шагов
              <HelpHint text={HH.steps} />
            </span>
            <input type="number" value={scenario.steps} onChange={(e) => setScenario({ ...scenario, steps: Number(e.target.value) })} />
          </label>
          <label>
            <span className="labelRow">
              Delta time (сек)
              <HelpHint text={HH.deltaTime} />
            </span>
            <input
              type="number"
              min={DELTA_TIME_MIN}
              max={DELTA_TIME_MAX}
              value={scenario.deltaTime}
              onChange={(e) => setScenario({ ...scenario, deltaTime: clamp(e.target.value, DELTA_TIME_MIN, DELTA_TIME_MAX) })}
            />
          </label>
          <label>
            <span className="labelRow">
              Setpoint (C)
              <HelpHint text={HH.setpoint} />
            </span>
            <input type="number" value={scenario.setpoint} onChange={(e) => setScenario({ ...scenario, setpoint: Number(e.target.value) })} />
          </label>
          <label>
            <span className="labelRow">
              Вентиляторы CRAC (%)
              <HelpHint text={HH.fanSpeed} />
            </span>
            <input type="number" value={scenario.fanSpeed} onChange={(e) => setScenario({ ...scenario, fanSpeed: Number(e.target.value) })} />
          </label>
          <label>
            <span className="labelRow">
              Режим охлаждения
              <HelpHint text={HH.coolingMode} />
            </span>
            <select value={scenario.coolingMode} onChange={(e) => setScenario({ ...scenario, coolingMode: e.target.value })}>
              <option value="free">free</option>
              <option value="chiller">chiller</option>
              <option value="mixed">mixed</option>
            </select>
          </label>
          <label>
            <span className="labelRow">
              Средняя нагрузка (0..1)
              <HelpHint text={HH.meanLoad} />
            </span>
            <input type="number" step="0.01" value={scenario.meanLoad} onChange={(e) => setScenario({ ...scenario, meanLoad: Number(e.target.value) })} />
          </label>
          <label>
            <span className="labelRow">
              Разброс нагрузки (std)
              <HelpHint text={HH.stdLoad} />
            </span>
            <input type="number" step="0.01" value={scenario.stdLoad} onChange={(e) => setScenario({ ...scenario, stdLoad: Number(e.target.value) })} />
          </label>
          <label>
            <span className="labelRow">
              Наружная температура (C)
              <HelpHint text={HH.outsideTemp} />
            </span>
            <input type="number" value={scenario.outsideTemp} onChange={(e) => setScenario({ ...scenario, outsideTemp: Number(e.target.value) })} />
          </label>
          <label>
            <span className="labelRow">
              Источник нагрузки
              <HelpHint text={HH.useDatasetLoad} />
            </span>
            <select
              value={scenario.useDatasetLoad ? 'dataset' : 'random'}
              onChange={(e) => setScenario({ ...scenario, useDatasetLoad: e.target.value === 'dataset' })}
            >
              <option value="random">Случайная (mean/std)</option>
              <option value="dataset">Из датасета</option>
            </select>
          </label>
          <label>
            <span className="labelRow">
              Датасет нагрузки
              <HelpHint text={HH.loadDataset} />
            </span>
            <select value={selectedLoadDataset} onChange={(e) => setSelectedLoadDataset(e.target.value)}>
              <option value="">выбрать</option>
              {datasets.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="labelRow">
              Загрузить CSV нагрузки
              <HelpHint text={HH.loadCsvUpload} />
            </span>
            <input type="file" accept=".csv" onChange={(e) => setLoadFile(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            <span className="labelRow">
              Режим реализма
              <HelpHint text={HH.realismMode} />
            </span>
            <input type="text" value="realistic (fixed)" disabled />
          </label>
          <label>
            <span className="labelRow">
              Динамический CRAC power
              <HelpHint text={HH.dynamicCrac} />
            </span>
            <select
              value={realism.use_dynamic_crac_power ? 'on' : 'off'}
              onChange={(e) => setRealism({ ...realism, use_dynamic_crac_power: e.target.value === 'on' })}
            >
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </label>
          <label>
            <span className="labelRow">
              Клип зала min
              <HelpHint text={HH.clipRoomMin} />
            </span>
            <input
              type="number"
              value={realism.room_temp_clip_min}
              onChange={(e) => setRealism({ ...realism, room_temp_clip_min: Number(e.target.value) })}
            />
          </label>
          <label>
            <span className="labelRow">
              Клип зала max
              <HelpHint text={HH.clipRoomMax} />
            </span>
            <input
              type="number"
              value={realism.room_temp_clip_max}
              onChange={(e) => setRealism({ ...realism, room_temp_clip_max: Number(e.target.value) })}
            />
          </label>
          <label>
            <span className="labelRow">
              Клип чипа multiplier
              <HelpHint text={HH.clipChip} />
            </span>
            <input
              type="number"
              step="0.1"
              value={realism.chip_temp_clip_multiplier}
              onChange={(e) => setRealism({ ...realism, chip_temp_clip_multiplier: Number(e.target.value) })}
            />
          </label>
        </div>
        <div className="actions">
          <button onClick={runFastScenario} disabled={progress.running}>
            Запустить быстрый прогон
          </button>
          <button onClick={runSingleStep} disabled={progress.running}>
            Сделать 1 шаг
          </button>
          <button onClick={uploadLoadDataset} disabled={progress.running}>
            Загрузить датасет нагрузки
          </button>
          <button onClick={applyRealism} disabled={progress.running}>
            Применить реализм
          </button>
          <button onClick={stopRun} disabled={!progress.running}>
            Остановить
          </button>
        </div>
        <div className="progressWrap">
          <div className="progressInfo">
            <span>
              Прогресс: {progress.step}/{progress.total}
            </span>
            <span>{progressPct}%</span>
          </div>
          <div className="progressBar">
            <div className="progressFill" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="hint">
            Безопасный диапазон delta time: {DELTA_TIME_MIN}-{DELTA_TIME_MAX} сек
          </div>
        </div>
      </section>

      {aggregates.length > 0 && (
        <section className="cards">
          {aggregates.map(([label, value]) => (
            <article key={label} className="card">
              <h3>{label}</h3>
              <p>{value}</p>
            </article>
          ))}
        </section>
      )}

      {servers.length > 0 && (
        <section className="rackPanel">
          <h2>Итоговое состояние серверов ({servers.length})</h2>
          <div className="rackGrid">
            {servers.map((server) => (
              <div key={server.server_id} className="serverCard">
                <strong>Сервер {server.server_id + 1}</strong>
                <span>Чип: {server.t_chip.toFixed(2)} C</span>
                <span>Вход: {server.t_in.toFixed(2)} C</span>
                <span>Выход: {server.t_out.toFixed(2)} C</span>
                <span>Нагрузка: {(server.utilization * 100).toFixed(1)}%</span>
                <span>Мощность: {server.power.toFixed(1)} W</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {chartData.length > 1 && (
        <section className="charts">
          <h2>Графики после прогона</h2>
          <div className="chartGrid">
            <div className="chartCard">
              <h3>Температуры</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#2b3a5c" strokeDasharray="3 3" />
                  <XAxis dataKey="timeMin" name="Время" unit=" мин" />
                  <YAxis domain={['auto', 'auto']} />
                  <Tooltip />
                  <Line type="monotone" dataKey="avgChip" stroke="#ff8c42" dot={false} strokeWidth={1.2} name="Средняя темп. чипа" />
                  <Line type="monotone" dataKey="avgChipMa" stroke="#ffd2b0" dot={false} strokeWidth={2} name="Чип (сглаж.)" />
                  <Line type="monotone" dataKey="room" stroke="#4d8bff" dot={false} name="Темп. зала" />
                  <Line type="monotone" dataKey="roomMa" stroke="#bcd6ff" dot={false} name="Зал (сглаж.)" />
                  <Line type="monotone" dataKey="outside" stroke="#9ec5ff" dot={false} name="Наружная темп." />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chartCard">
              <h3>PUE и риск перегрева</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#2b3a5c" strokeDasharray="3 3" />
                  <XAxis dataKey="timeMin" />
                  <YAxis domain={[1, 'auto']} />
                  <Tooltip />
                  <Line type="monotone" dataKey="pue" stroke="#50e3a4" dot={false} strokeWidth={1.2} name="PUE" />
                  <Line type="monotone" dataKey="pueMa" stroke="#b8ffd9" dot={false} strokeWidth={2} name="PUE (сглаж.)" />
                  <Line type="monotone" dataKey="overheatRisk" stroke="#ff4d6d" dot={false} name="Риск перегрева, %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chartCard">
              <h3>Мощность</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#2b3a5c" strokeDasharray="3 3" />
                  <XAxis dataKey="timeMin" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="totalPowerKw" stroke="#62f0ff" dot={false} strokeWidth={1.2} name="Серверы, кВт" />
                  <Line type="monotone" dataKey="coolingPower" stroke="#3a66ff" dot={false} strokeWidth={1.2} name="Охлаждение, Вт" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
