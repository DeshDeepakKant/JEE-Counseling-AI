document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.feature-card');
    const buttons = document.querySelectorAll('.btn');

    // Add staggered animation to cards
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
        
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 400 + (index * 150));
    });

    // Add ripple effect or simple click feedback to buttons
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            if (btn.getAttribute('href') === '#') {
                e.preventDefault();
                console.log(`${btn.textContent} clicked!`);
                
                // Temporary click feedback
                btn.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    btn.style.transform = '';
                }, 100);
            }
        });
    });

    // Parallax-ish effect on container based on mouse movement
    const container = document.querySelector('.container');
    document.addEventListener('mousemove', (e) => {
        const xAxis = (window.innerWidth / 2 - e.pageX) / 50;
        const yAxis = (window.innerHeight / 2 - e.pageY) / 50;
        // container.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
        // Subtle move instead of rotation to keep it clean
        container.style.boxShadow = `${xAxis}px ${yAxis}px 50px -12px rgba(0, 0, 0, 0.5)`;
    });
});
