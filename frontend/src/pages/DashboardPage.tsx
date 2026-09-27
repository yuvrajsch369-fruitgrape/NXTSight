import FeatureGrid from '@/components/FeatureGrid'
import Hero from '@/components/Hero'
import StatusBadges from '@/components/StatusBadges'
import VisionCallout from '@/components/VisionCallout'

export default function DashboardPage() {
  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-10 px-6 pb-24">
      <Hero />
      <StatusBadges />
      <VisionCallout />
      <FeatureGrid />
    </main>
  )
}
