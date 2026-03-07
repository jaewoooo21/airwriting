
        // Set default chart fonts to be more elegant
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.color = '#94A3B8';

        // 1. Accuracy Chart (Bar Chart with Gradient)
        const ctxAcc = document.getElementById('accuracyChart');
        if (ctxAcc) {
            const gradBtn = ctxAcc.getContext('2d').createLinearGradient(0, 0, 0, 400);
            gradBtn.addColorStop(0, '#38BDF8');
            gradBtn.addColorStop(1, 'rgba(56, 189, 248, 0.1)');

            new Chart(ctxAcc, {
                type: 'bar',
                data: {
                    labels: ['Double Integration (Acc Only)', 'ESKF (6-DOF)', 'MARG (9-DOF)', 'MARG + ZUPT', 'Our FK Model (0-Drift)', 'RF ML Model'],
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: [55.3, 72.1, 85.0, 93.5, 96.2, 98.1],
                        backgroundColor: gradBtn,
                        borderRadius: 6,
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: { callback: function (val) { return val + '%' } }
                        },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        // 2. Drift Error Chart (Line Chart with Fill & Glow)
        const ctxDrift = document.getElementById('driftChart');
        if (ctxDrift) {
            const gradDrift1 = ctxDrift.getContext('2d').createLinearGradient(0, 0, 0, 400);
            gradDrift1.addColorStop(0, 'rgba(239, 68, 68, 0.5)'); // Red glow for high error
            gradDrift1.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

            const gradDrift2 = ctxDrift.getContext('2d').createLinearGradient(0, 0, 0, 400);
            gradDrift2.addColorStop(0, 'rgba(16, 185, 129, 0.5)'); // Green glow for low error
            gradDrift2.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

            // Mock Data: 60 seconds
            const timeLabels = Array.from({ length: 13 }, (_, i) => i * 5); // 0, 5, 10 ... 60
            const eskeError = timeLabels.map(t => 0.01 * (t * t / 20)); // Exponential drift
            const fkError = timeLabels.map(t => 0.02 + Math.random() * 0.015); // Stable bounded drift

            new Chart(ctxDrift, {
                type: 'line',
                data: {
                    labels: timeLabels.map(t => t + 's'),
                    datasets: [
                        {
                            label: 'Standard Integration Drift',
                            data: eskeError,
                            borderColor: '#EF4444',
                            backgroundColor: gradDrift1,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0
                        },
                        {
                            label: 'Ours (FK + SHOE ZUPT)',
                            data: fkError,
                            borderColor: '#10B981',
                            backgroundColor: gradDrift2,
                            fill: true,
                            tension: 0.4,
                            borderWidth: 3,
                            pointRadius: 3,
                            pointBackgroundColor: '#fff'
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Positional Error (m)' },
                            grid: { color: 'rgba(255,255,255,0.05)' }
                        },
                        x: { grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            });
        }

        // Additional Charts (Ablation & Comparison)
        const ctxAblation = document.getElementById('ablationChart');
        if (ctxAblation) {
            const gradAblation = ctxAblation.getContext('2d').createLinearGradient(0, 0, 800, 0); // Horizontal gradient for horizontal bar chart
            gradAblation.addColorStop(0, 'rgba(139, 92, 246, 0.8)'); // Purple glow
            gradAblation.addColorStop(1, 'rgba(56, 189, 248, 0.8)'); // Blue glow

            new Chart(ctxAblation, {
                type: 'bar',
                data: {
                    labels: ['Full System', '- ZUPT', '- MARG (6-axis only)', '- Writing Plane', '- FK (ESKF only)', '- All Corrections'],
                    datasets: [{
                        label: 'Recognition Accuracy %',
                        data: [98.1, 91.2, 89.5, 95.3, 85.2, 72.0],
                        backgroundColor: function (context) {
                            // The first bar (Full System) gets the nice gradient. Others get a muted color.
                            return context.dataIndex === 0 ? gradAblation : 'rgba(255, 255, 255, 0.1)';
                        },
                        borderColor: function (context) {
                            return context.dataIndex === 0 ? '#38BDF8' : 'rgba(255, 255, 255, 0.2)';
                        },
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: {
                            min: 60, max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                            ticks: { callback: function (val) { return val + '%' } }
                        },
                        y: { grid: { display: false }, ticks: { color: '#e2e8f0', font: { size: 12, family: "'JetBrains Mono', monospace" } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        const ctxComp = document.getElementById('comparisonChart');
        if (ctxComp) {
            new Chart(ctxComp, {
                type: 'radar',
                data: {
                    labels: ['정확도', '지연시간', '비용효율', '휴대성', '설치 편의성', '드리프트 억제'],
                    datasets: [
                        {
                            label: 'IMU-Only FK (Ours)',
                            data: [98, 95, 100, 95, 90, 85],
                            borderColor: '#10B981', // Emerald
                            backgroundColor: 'rgba(16, 185, 129, 0.2)',
                            pointBackgroundColor: '#10B981',
                            pointBorderColor: '#fff',
                            borderWidth: 2
                        },
                        {
                            label: 'UWB + IMU',
                            data: [99, 85, 50, 60, 40, 95],
                            borderColor: '#64748b',
                            backgroundColor: 'rgba(100, 116, 139, 0.1)',
                            pointBackgroundColor: '#64748b',
                            borderDash: [5, 5],
                            borderWidth: 2
                        },
                        {
                            label: 'Camera-Based',
                            data: [97, 70, 30, 40, 50, 100],
                            borderColor: '#334155',
                            backgroundColor: 'rgba(51, 65, 85, 0.05)',
                            pointBackgroundColor: '#334155',
                            borderDash: [2, 2],
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        r: {
                            min: 0, max: 100,
                            ticks: { stepSize: 25, color: '#64748b', backdropColor: 'transparent' },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            pointLabels: { color: '#cbd5e1', font: { size: 12, family: "'Inter', sans-serif" } },
                            angleLines: { color: 'rgba(255, 255, 255, 0.05)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#e2e8f0', usePointStyle: true, font: { family: "'Inter', sans-serif" } } }
                    }
                }
            });
        }
    