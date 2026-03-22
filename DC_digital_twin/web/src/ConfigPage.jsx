import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import { HelpHint, LabelRow } from './HelpHint.jsx'
import { CFG_HINTS as C } from './hintsRu.js'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

/** Базовый URL секции /config/ (со слэшем — иначе FastAPI отдаёт 404). */
const configApiRoot = () => `${API_BASE.replace(/\/$/, '')}/config/`

function Num({ label, value, onChange, step, hint }) {
  return (
    <label>
      <LabelRow hint={hint}>{label}</LabelRow>
      <input type="number" step={step ?? 'any'} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

function Txt({ label, value, onChange, hint }) {
  return (
    <label>
      <LabelRow hint={hint}>{label}</LabelRow>
      <input type="text" value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}

function Bool({ label, value, onChange, hint }) {
  return (
    <label>
      <LabelRow hint={hint}>{label}</LabelRow>
      <select value={value ? 'yes' : 'no'} onChange={(e) => onChange(e.target.value === 'yes')}>
        <option value="yes">да</option>
        <option value="no">нет</option>
      </select>
    </label>
  )
}

export default function ConfigPage() {
  const [config, setConfig] = useState(null)
  const [jsonMode, setJsonMode] = useState(false)
  const [jsonText, setJsonText] = useState('')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)

  const loadFromServer = useCallback(async () => {
    setError('')
    const { data } = await axios.get(configApiRoot())
    setConfig(data)
    setJsonText(JSON.stringify(data, null, 2))
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        await loadFromServer()
      } catch (e) {
        if (!cancelled) setError(String(e?.response?.data?.detail ?? e?.message ?? e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [loadFromServer])

  const loadDefaults = async () => {
    setError('')
    try {
      const { data } = await axios.get(`${configApiRoot()}defaults`)
      setConfig(data)
      setJsonText(JSON.stringify(data, null, 2))
      setMessage('Подставлены значения по умолчанию (ещё не сохранено на сервер)')
    } catch (e) {
      setError(String(e?.response?.data?.detail ?? e?.message ?? e))
    }
  }

  const save = async () => {
    setError('')
    setMessage('')
    try {
      let payload = config
      if (jsonMode) {
        payload = JSON.parse(jsonText)
      }
      const { data } = await axios.put(configApiRoot(), payload)
      setMessage(data?.message ?? 'Сохранено')
      await loadFromServer()
    } catch (e) {
      setError(e?.response?.data?.detail ?? (e instanceof SyntaxError ? 'Неверный JSON' : String(e)))
    }
  }

  if (loading || !config) {
    return (
      <div className="layout">
        <p>Загрузка конфигурации…</p>
        {error && <div className="error">{error}</div>}
      </div>
    )
  }

  const profName = config.servers?.default_profile ?? 'dl380'
  const profile = config.servers?.profiles?.[profName] ?? {}

  const setSim = (patch) => setConfig((c) => ({ ...c, simulator: { ...c.simulator, ...patch } }))
  const setRealism = (patch) => setConfig((c) => ({ ...c, realism: { ...c.realism, ...patch } }))
  const setRack = (patch) => setConfig((c) => ({ ...c, rack: { ...c.rack, ...patch } }))
  const setServersTop = (patch) => setConfig((c) => ({ ...c, servers: { ...c.servers, ...patch } }))
  const setProfile = (patch) =>
    setConfig((c) => ({
      ...c,
      servers: {
        ...c.servers,
        profiles: {
          ...c.servers.profiles,
          [profName]: { ...profile, ...patch },
        },
      },
    }))
  const setCrac = (patch) =>
    setConfig((c) => ({
      ...c,
      cooling: {
        ...c.cooling,
        crac: { ...c.cooling.crac, ...patch },
      },
    }))
  const setFans = (patch) =>
    setConfig((c) => ({
      ...c,
      cooling: {
        ...c.cooling,
        fans: { ...c.cooling.fans, ...patch },
      },
    }))
  const setRoom = (patch) => setConfig((c) => ({ ...c, room: { ...c.room, ...patch } }))
  const setLoadGen = (patch) => setConfig((c) => ({ ...c, load_generator: { ...c.load_generator, ...patch } }))
  const setMqtt = (patch) => setConfig((c) => ({ ...c, mqtt: { ...c.mqtt, ...patch } }))
  const setOutput = (patch) => setConfig((c) => ({ ...c, output: { ...c.output, ...patch } }))
  const setLogging = (patch) => setConfig((c) => ({ ...c, logging: { ...c.logging, ...patch } }))
  const setWeather = (patch) =>
    setConfig((c) => ({
      ...c,
      weather: { ...(c.weather ?? { enabled: false }), ...patch },
    }))

  const cop = config.cooling?.crac?.cop_curve ?? [0, 0, 0]

  return (
    <div className="layout">
      <header className="topbar">
        <h1>Полная конфигурация симулятора</h1>
        <nav className="topnav">
          <Link className="navLink" to="/">
            ← Прогон
          </Link>
        </nav>
      </header>

      {error && <div className="error">{error}</div>}
      {message && <div className="hint" style={{ marginBottom: 8 }}>{message}</div>}

      <section className="quickPanel">
        <div className="actions" style={{ marginBottom: 12 }}>
          <button
            type="button"
            onClick={() => {
              if (!jsonMode) {
                setJsonText(JSON.stringify(config, null, 2))
              }
              setJsonMode((v) => !v)
            }}
          >
            {jsonMode ? 'Форма' : 'JSON'}
          </button>
          <button type="button" onClick={loadFromServer}>
            Обновить с сервера
          </button>
          <button type="button" onClick={loadDefaults}>
            Значения по умолчанию
          </button>
          <button type="button" onClick={save}>
            Сохранить на сервер
          </button>
        </div>
        <p className="hint">
          После сохранения симулятор пересоздаётся (сбрасываются шаги и история). Не сохраняйте во время realtime/fast прогона.
        </p>
      </section>

      {jsonMode ? (
        <section className="quickPanel">
          <h2 className="labelRow" style={{ alignItems: 'center', gap: 8 }}>
            JSON
            <HelpHint text={C.json_block} />
          </h2>
          <textarea
            className="jsonArea"
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            spellCheck={false}
            rows={32}
            style={{ width: '100%', fontFamily: 'monospace', fontSize: 12 }}
          />
        </section>
      ) : (
        <>
          <section className="quickPanel">
            <h2>Симулятор</h2>
            <div className="controlGrid">
              <Txt label="Имя" hint={C.sim_name} value={config.simulator?.name ?? ''} onChange={(v) => setSim({ name: v })} />
              <Num label="time_step (с)" hint={C.sim_time_step} value={config.simulator?.time_step ?? 1} onChange={(v) => setSim({ time_step: v })} />
              <Num label="realtime_factor" hint={C.sim_realtime_factor} value={config.simulator?.realtime_factor ?? 1} onChange={(v) => setSim({ realtime_factor: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Реализм</h2>
            <div className="controlGrid">
              <Txt label="mode" hint={C.real_mode} value={config.realism?.mode ?? 'realistic'} onChange={(v) => setRealism({ mode: v })} />
              <Bool
                label="use_dynamic_crac_power"
                hint={C.real_dynamic_crac}
                value={!!config.realism?.use_dynamic_crac_power}
                onChange={(v) => setRealism({ use_dynamic_crac_power: v })}
              />
              <Num label="room_temp_clip_min" hint={C.real_clip_min} value={config.realism?.room_temp_clip_min ?? 10} onChange={(v) => setRealism({ room_temp_clip_min: v })} />
              <Num label="room_temp_clip_max" hint={C.real_clip_max} value={config.realism?.room_temp_clip_max ?? 40} onChange={(v) => setRealism({ room_temp_clip_max: v })} />
              <Num
                label="chip_temp_clip_multiplier"
                hint={C.real_chip_mult}
                value={config.realism?.chip_temp_clip_multiplier ?? 1.2}
                step="0.01"
                onChange={(v) => setRealism({ chip_temp_clip_multiplier: v })}
              />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Стойка</h2>
            <div className="controlGrid">
              <Num label="height (м)" hint={C.rack_h} value={config.rack?.height ?? 2} onChange={(v) => setRack({ height: v })} />
              <Num label="width" hint={C.rack_w} value={config.rack?.width ?? 0.6} onChange={(v) => setRack({ width: v })} />
              <Num label="depth" hint={C.rack_d} value={config.rack?.depth ?? 1} onChange={(v) => setRack({ depth: v })} />
              <Num label="num_units" hint={C.rack_units} value={config.rack?.num_units ?? 42} onChange={(v) => setRack({ num_units: v })} />
              <label>
                <LabelRow hint={C.rack_containment}>containment</LabelRow>
                <select
                  value={config.rack?.containment ?? 'none'}
                  onChange={(e) => setRack({ containment: e.target.value })}
                >
                  <option value="none">none</option>
                  <option value="hot_aisle">hot_aisle</option>
                  <option value="cold_aisle">cold_aisle</option>
                </select>
              </label>
            </div>
          </section>

          <section className="quickPanel">
            <h2>Серверы</h2>
            <div className="controlGrid">
              <Txt label="default_profile" hint={C.srv_profile} value={config.servers?.default_profile ?? 'dl380'} onChange={(v) => setServersTop({ default_profile: v })} />
              <Num label="count" hint={C.srv_count} value={config.servers?.count ?? 1} onChange={(v) => setServersTop({ count: Math.max(1, Math.floor(v)) })} />
              <Txt label="arrangement" hint={C.srv_arr} value={config.servers?.arrangement ?? 'uniform'} onChange={(v) => setServersTop({ arrangement: v })} />
              <h3 style={{ gridColumn: '1 / -1' }}>Профиль «{profName}»</h3>
              <Num label="p_idle (Вт)" hint={C.srv_p_idle} value={profile.p_idle ?? 0} onChange={(v) => setProfile({ p_idle: v })} />
              <Num label="p_max (Вт)" hint={C.srv_p_max} value={profile.p_max ?? 0} onChange={(v) => setProfile({ p_max: v })} />
              <Num label="t_max (°C)" hint={C.srv_t_max} value={profile.t_max ?? 85} onChange={(v) => setProfile({ t_max: v })} />
              <Num label="c_thermal" hint={C.srv_c_thermal} value={profile.c_thermal ?? 5000} onChange={(v) => setProfile({ c_thermal: v })} />
              <Num label="m_dot" hint={C.srv_m_dot} value={profile.m_dot ?? 0.05} step="0.001" onChange={(v) => setProfile({ m_dot: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Охлаждение</h2>
            <div className="controlGrid">
              <Txt label="crac.type" hint={C.crac_type} value={config.cooling?.crac?.type ?? 'chilled_water'} onChange={(v) => setCrac({ type: v })} />
              <Num label="capacity (Вт)" hint={C.crac_cap} value={config.cooling?.crac?.capacity ?? 30000} onChange={(v) => setCrac({ capacity: v })} />
              <Num label="default_setpoint" hint={C.crac_setpoint} value={config.cooling?.crac?.default_setpoint ?? 22} onChange={(v) => setCrac({ default_setpoint: v })} />
              <Num label="airflow_m_dot" hint={C.crac_airflow} value={config.cooling?.crac?.airflow_m_dot ?? 2} step="0.01" onChange={(v) => setCrac({ airflow_m_dot: v })} />
              <Num label="supply_approach" hint={C.crac_approach} value={config.cooling?.crac?.supply_approach ?? 1} step="0.1" onChange={(v) => setCrac({ supply_approach: v })} />
              <Txt label="COP a,b,c (через запятую)" hint={C.crac_cop} value={cop.join(',')} onChange={(v) => setCrac({ cop_curve: v.split(',').map((x) => Number(x.trim())) })} />
              <Num label="fans.max_power" hint={C.fans_max} value={config.cooling?.fans?.max_power ?? 2000} onChange={(v) => setFans({ max_power: v })} />
              <label>
                <LabelRow hint={C.fans_law}>fans.law</LabelRow>
                <select value={config.cooling?.fans?.law ?? 'cubic'} onChange={(e) => setFans({ law: e.target.value })}>
                  <option value="cubic">cubic</option>
                  <option value="linear">linear</option>
                </select>
              </label>
            </div>
          </section>

          <section className="quickPanel">
            <h2>Помещение</h2>
            <div className="controlGrid">
              <Num label="volume (м³)" hint={C.room_vol} value={config.room?.volume ?? 100} onChange={(v) => setRoom({ volume: v })} />
              <Num label="wall_heat_transfer (Вт/К)" hint={C.room_wall} value={config.room?.wall_heat_transfer ?? 12} onChange={(v) => setRoom({ wall_heat_transfer: v })} />
              <Num
                label="supply_mixing_conductance (Вт/К)"
                hint={C.room_supply_g}
                value={config.room?.supply_mixing_conductance ?? 14}
                onChange={(v) => setRoom({ supply_mixing_conductance: v })}
              />
              <Num label="initial_temperature" hint={C.room_t0} value={config.room?.initial_temperature ?? 22} onChange={(v) => setRoom({ initial_temperature: v })} />
              <Num label="thermal_mass_factor" hint={C.room_mass} value={config.room?.thermal_mass_factor ?? 10} onChange={(v) => setRoom({ thermal_mass_factor: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Генератор нагрузки</h2>
            <div className="controlGrid">
              <label>
                <LabelRow hint={C.lg_type}>type</LabelRow>
                <select value={config.load_generator?.type ?? 'random'} onChange={(e) => setLoadGen({ type: e.target.value })}>
                  <option value="random">random</option>
                  <option value="constant">constant</option>
                  <option value="periodic">periodic</option>
                  <option value="dataset">dataset</option>
                </select>
              </label>
              <Txt label="dataset_path" hint={C.lg_path} value={config.load_generator?.dataset_path ?? ''} onChange={(v) => setLoadGen({ dataset_path: v })} />
              <Num label="random_seed" hint={C.lg_seed} value={config.load_generator?.random_seed ?? 42} onChange={(v) => setLoadGen({ random_seed: v })} />
              <Num label="mean_load" hint={C.lg_mean} value={config.load_generator?.mean_load ?? 0.6} step="0.01" onChange={(v) => setLoadGen({ mean_load: v })} />
              <Num label="std_load" hint={C.lg_std} value={config.load_generator?.std_load ?? 0.2} step="0.01" onChange={(v) => setLoadGen({ std_load: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>MQTT</h2>
            <div className="controlGrid">
              <Bool label="enabled" hint={C.mqtt_en} value={!!config.mqtt?.enabled} onChange={(v) => setMqtt({ enabled: v })} />
              <Txt label="broker" hint={C.mqtt_broker} value={config.mqtt?.broker ?? 'localhost'} onChange={(v) => setMqtt({ broker: v })} />
              <Num label="port" hint={C.mqtt_port} value={config.mqtt?.port ?? 1883} onChange={(v) => setMqtt({ port: v })} />
              <Txt label="topic_prefix" hint={C.mqtt_prefix} value={config.mqtt?.topic_prefix ?? ''} onChange={(v) => setMqtt({ topic_prefix: v })} />
              <Num label="publish_rate" hint={C.mqtt_rate} value={config.mqtt?.publish_rate ?? 1} step="0.1" onChange={(v) => setMqtt({ publish_rate: v })} />
              <Num label="qos" hint={C.mqtt_qos} value={config.mqtt?.qos ?? 1} onChange={(v) => setMqtt({ qos: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Вывод</h2>
            <div className="controlGrid">
              <Bool label="enabled" hint={C.out_en} value={!!config.output?.enabled} onChange={(v) => setOutput({ enabled: v })} />
              <label>
                <LabelRow hint={C.out_fmt}>format</LabelRow>
                <select value={config.output?.format ?? 'csv'} onChange={(e) => setOutput({ format: e.target.value })}>
                  <option value="csv">csv</option>
                  <option value="json">json</option>
                  <option value="parquet">parquet</option>
                </select>
              </label>
              <Txt label="path" hint={C.out_path} value={config.output?.path ?? 'results'} onChange={(v) => setOutput({ path: v })} />
              <Num label="save_interval" hint={C.out_interval} value={config.output?.save_interval ?? 100} onChange={(v) => setOutput({ save_interval: v })} />
              <Bool label="compression" hint={C.out_zip} value={!!config.output?.compression} onChange={(v) => setOutput({ compression: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Логирование</h2>
            <div className="controlGrid">
              <label>
                <LabelRow hint={C.log_level}>level</LabelRow>
                <select value={config.logging?.level ?? 'INFO'} onChange={(e) => setLogging({ level: e.target.value })}>
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </select>
              </label>
              <label>
                <LabelRow hint={C.log_fmt}>format</LabelRow>
                <select value={config.logging?.format ?? 'json'} onChange={(e) => setLogging({ format: e.target.value })}>
                  <option value="json">json</option>
                  <option value="text">text</option>
                </select>
              </label>
              <Txt label="file" hint={C.log_file} value={config.logging?.file ?? ''} onChange={(v) => setLogging({ file: v })} />
            </div>
          </section>

          <section className="quickPanel">
            <h2>Погода (опционально)</h2>
            <div className="controlGrid">
              <Bool label="weather.enabled" hint={C.weather_en} value={!!config.weather?.enabled} onChange={(v) => setWeather({ enabled: v })} />
            </div>
          </section>
        </>
      )}
    </div>
  )
}
