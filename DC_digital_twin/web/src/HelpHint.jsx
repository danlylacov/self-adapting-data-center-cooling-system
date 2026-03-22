/**
 * Знак «?» с всплывающей подсказкой при наведении (и фокусе для клавиатуры).
 */
export function HelpHint({ text }) {
  if (!text) return null
  return (
    <span className="helpHint" tabIndex={0} aria-label={text}>
      <span className="helpHintMark" aria-hidden>
        ?
      </span>
      <span className="helpHintTooltip" role="tooltip">
        {text}
      </span>
    </span>
  )
}

/** Подпись поля + «?» в одной строке */
export function LabelRow({ children, hint }) {
  return (
    <span className="labelRow">
      {children}
      {hint ? <HelpHint text={hint} /> : null}
    </span>
  )
}
