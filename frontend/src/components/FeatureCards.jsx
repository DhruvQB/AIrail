export default function FeatureCards({ onCardClick }) {
  const cards = [
    {
      icon: '🚄',
      title: 'Live Train Status',
      desc: 'Track any train in real-time with delay info and current position.',
      prompt: 'What is the live status of train 12951?',
    },
    {
      icon: '🎫',
      title: 'PNR Status Check',
      desc: 'Check your booking status, waitlist position, and chart details.',
      prompt: 'Check PNR status for 8348138555',
    },
    {
      icon: '📋',
      title: 'Railway Policies',
      desc: 'Get info on cancellation charges, refund rules, and tatkal booking.',
      prompt: 'What are the cancellation charges for confirmed tickets?',
    },
  ];

  return (
    <div className="feature-cards">
      {cards.map((card, i) => (
        <div
          key={i}
          className="feature-card"
          onClick={() => onCardClick(card.prompt)}
        >
          <span className="feature-card-icon">{card.icon}</span>
          <div className="feature-card-title">{card.title}</div>
          <div className="feature-card-desc">{card.desc}</div>
        </div>
      ))}
    </div>
  );
}
