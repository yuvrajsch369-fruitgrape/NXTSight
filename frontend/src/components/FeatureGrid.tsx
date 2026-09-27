import { FEATURES } from '@/lib/features'
import FeatureCard from './FeatureCard'

export default function FeatureGrid() {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
      {FEATURES.map((feature, i) => (
        <FeatureCard key={feature.name} feature={feature} index={i} />
      ))}
    </div>
  )
}
