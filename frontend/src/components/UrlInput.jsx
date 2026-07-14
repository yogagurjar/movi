export default function UrlInput({ label, placeholder, value, onChange, icon }) {
  return (
    <div className="group">
      <label className="block text-xs font-medium text-cinema-300 mb-2 tracking-wide uppercase">
        {icon} {label}
      </label>
      <div className="relative">
        <input
          type="url"
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-4 py-3.5 rounded-xl bg-cinema-800 border border-cinema-600 text-white placeholder-cinema-500 text-sm
                     focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50 transition-all duration-200
                     group-hover:border-cinema-500"
        />
        {value && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="w-2 h-2 rounded-full bg-green-500" />
          </div>
        )}
      </div>
    </div>
  )
}
