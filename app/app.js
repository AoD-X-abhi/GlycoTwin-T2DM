/* ==========================================================================
   GlycoTwin Mobile Digital Twin - Application Logic & Engine
   ========================================================================== */

// Fallback cohort profiles for local file:// opening without web server
const FALLBACK_PROFILES = {
  "11": {
    id: 11,
    label: "Subject 011 (Pre-DM)",
    group: "Pre-diabetes",
    age: 49,
    gender: "F",
    bmi: 27.4,
    hba1c: 6.1,
    fasting_glu: 106.0,
    insulin: 13.5,
    current_glucose: 118.0,
    tir_daily: 89.0,
    heart_rate: 72,
    current_mets: 1.2,
    microbiome_vae: [0.124, -0.452, 0.789, 0.045, -0.321, 0.654, -0.118, 0.342],
    microbial_richness: 412,
    butyrate_score: 4,
    gut_health_score: 82,
    sample_meals: [
      { name: "Morning Oatmeal & Berries", carbs: 55, protein: 18, fat: 12, fiber: 7, calories: 420 },
      { name: "Chicken & Quinoa Bowl", carbs: 62, protein: 32, fat: 16, fiber: 8, calories: 540 },
      { name: "Whole Wheat Pasta & Sauce", carbs: 78, protein: 22, fat: 14, fiber: 6, calories: 560 },
      { name: "Avocado & Sourdough Toast", carbs: 42, protein: 14, fat: 22, fiber: 9, calories: 440 }
    ]
  },
  "45": {
    id: 45,
    label: "Subject 045 (Type 2 DM)",
    group: "Type 2 Diabetes",
    age: 56,
    gender: "M",
    bmi: 31.8,
    hba1c: 7.8,
    fasting_glu: 148.0,
    insulin: 22.4,
    current_glucose: 152.0,
    tir_daily: 68.0,
    heart_rate: 78,
    current_mets: 1.1,
    microbiome_vae: [-0.642, 0.812, -0.345, 0.764, 0.221, -0.589, 0.432, -0.215],
    microbial_richness: 295,
    butyrate_score: 2,
    gut_health_score: 58,
    sample_meals: [
      { name: "White Rice & Curry Beef", carbs: 88, protein: 28, fat: 24, fiber: 3, calories: 720 },
      { name: "Bagel with Cream Cheese", carbs: 68, protein: 12, fat: 18, fiber: 2, calories: 490 },
      { name: "Chicken Fried Rice", carbs: 82, protein: 26, fat: 22, fiber: 4, calories: 650 },
      { name: "Lentil Soup & Flatbread", carbs: 64, protein: 22, fat: 10, fiber: 11, calories: 450 }
    ]
  },
  "2": {
    id: 2,
    label: "Subject 002 (Healthy)",
    group: "Healthy Control",
    age: 34,
    gender: "F",
    bmi: 22.1,
    hba1c: 5.1,
    fasting_glu: 86.0,
    insulin: 6.2,
    current_glucose: 94.0,
    tir_daily: 99.0,
    heart_rate: 65,
    current_mets: 1.4,
    microbiome_vae: [0.551, 0.324, 0.612, -0.421, 0.814, 0.212, 0.543, 0.765],
    microbial_richness: 540,
    butyrate_score: 5,
    gut_health_score: 95,
    sample_meals: [
      { name: "Greek Yogurt & Granola", carbs: 45, protein: 24, fat: 10, fiber: 6, calories: 370 },
      { name: "Grilled Salmon & Veggies", carbs: 28, protein: 38, fat: 18, fiber: 8, calories: 440 },
      { name: "Mediterranean Hummus Bowl", carbs: 52, protein: 20, fat: 16, fiber: 12, calories: 460 },
      { name: "Berry Protein Smoothie", carbs: 38, protein: 30, fat: 6, fiber: 8, calories: 330 }
    ]
  },
  "22": {
    id: 22,
    label: "Subject 022 (High IR)",
    group: "Pre-diabetes",
    age: 52,
    gender: "M",
    bmi: 29.6,
    hba1c: 6.3,
    fasting_glu: 112.0,
    insulin: 18.0,
    current_glucose: 126.0,
    tir_daily: 82.0,
    heart_rate: 74,
    current_mets: 1.2,
    microbiome_vae: [-0.214, 0.342, 0.156, 0.452, -0.612, 0.125, -0.341, 0.118],
    microbial_richness: 375,
    butyrate_score: 3,
    gut_health_score: 71,
    sample_meals: [
      { name: "Pepperoni Pizza Slices", carbs: 74, protein: 26, fat: 28, fiber: 4, calories: 680 },
      { name: "Burrito Bowl with Carnitas", carbs: 70, protein: 34, fat: 24, fiber: 9, calories: 660 },
      { name: "Turkey Sub on Brioche", carbs: 58, protein: 28, fat: 14, fiber: 3, calories: 480 },
      { name: "Cheeseburger & Baked Fries", carbs: 65, protein: 32, fat: 30, fiber: 5, calories: 710 }
    ]
  }
};

class GlycoTwinApp {
  constructor() {
    this.profiles = FALLBACK_PROFILES;
    this.activeSubjectId = "11";
    this.activeProfile = this.profiles[this.activeSubjectId];

    // Current meal state
    this.currentMeal = {
      name: "Custom Meal",
      carbs: 55,
      fiber: 7,
      protein: 24,
      fat: 16,
      calories: 460
    };

    // Simulator intervention controls state
    this.simControls = {
      carbReductionPct: 50,
      walkMins: 30,
      walkMets: 3.5
    };

    // Scrubber state for trajectory canvas
    this.scrubberMinute = null;

    this.init();
  }

  async init() {
    this.setupClock();
    this.setupTabs();
    this.setupPatientSelect();
    this.setupMealControls();
    this.setupSimulatorControls();
    await this.loadRemoteCohortData();
    this.renderActiveProfile();
    this.renderPresets();
    this.renderSimulation();
    this.renderMicrobiomeRadar();
    this.drawSparkline();
  }

  setupClock() {
    const updateTime = () => {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, '0');
      const m = String(now.getMinutes()).padStart(2, '0');
      const clockEl = document.getElementById('live-clock');
      if (clockEl) clockEl.textContent = `${h}:${m}`;
    };
    updateTime();
    setInterval(updateTime, 30000);
  }

  async loadRemoteCohortData() {
    try {
      const res = await fetch('data/cohort_profiles.json');
      if (res.ok) {
        const data = await res.json();
        this.profiles = { ...FALLBACK_PROFILES, ...data };
        console.log("Remote cohort profiles loaded successfully.");
      }
    } catch (e) {
      console.log("Using built-in cohort fallback data.");
    }
  }

  setupTabs() {
    const tabBtns = document.querySelectorAll('.nav-tab-btn');
    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        this.switchTab(targetTab);
      });
    });

    document.getElementById('btnGoToMeal')?.addEventListener('click', () => {
      this.switchTab('screen-meals');
    });

    document.getElementById('btnRunSimulation')?.addEventListener('click', () => {
      this.switchTab('screen-simulator');
      this.renderSimulation();
    });
  }

  switchTab(tabId) {
    document.querySelectorAll('.app-screen').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-tab-btn').forEach(b => b.classList.remove('active'));

    const screen = document.getElementById(tabId);
    if (screen) screen.classList.add('active');

    const activeBtn = document.querySelector(`.nav-tab-btn[data-tab="${tabId}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    // Canvas redrawing upon visibility
    if (tabId === 'screen-simulator') {
      setTimeout(() => this.renderSimulation(), 50);
    } else if (tabId === 'screen-microbiome') {
      setTimeout(() => this.renderMicrobiomeRadar(), 50);
    } else if (tabId === 'screen-dashboard') {
      setTimeout(() => this.drawSparkline(), 50);
    }
  }

  setupPatientSelect() {
    const select = document.getElementById('patientSelect');
    if (!select) return;

    select.addEventListener('change', (e) => {
      this.activeSubjectId = e.target.value;
      this.activeProfile = this.profiles[this.activeSubjectId] || FALLBACK_PROFILES[this.activeSubjectId];
      this.renderActiveProfile();
      this.renderPresets();
      this.renderSimulation();
      this.renderMicrobiomeRadar();
      this.drawSparkline();
    });
  }

  setupMealControls() {
    const updateMacros = () => {
      const carbs = parseFloat(document.getElementById('inputCarbs').value);
      const fiber = parseFloat(document.getElementById('inputFiber').value);
      const protein = parseFloat(document.getElementById('inputProtein').value);
      const fat = parseFloat(document.getElementById('inputFat').value);

      document.getElementById('valCarbs').textContent = `${carbs} g`;
      document.getElementById('valFiber').textContent = `${fiber} g`;
      document.getElementById('valProtein').textContent = `${protein} g`;
      document.getElementById('valFat').textContent = `${fat} g`;

      const cals = Math.round((carbs * 4) + (protein * 4) + (fat * 9));
      document.getElementById('totalCaloriesVal').textContent = `${cals} kcal`;

      this.currentMeal = {
        name: this.currentMeal.name || "Custom Meal",
        carbs, fiber, protein, fat, calories: cals
      };
    };

    ['inputCarbs', 'inputFiber', 'inputProtein', 'inputFat'].forEach(id => {
      document.getElementById(id)?.addEventListener('input', updateMacros);
    });
  }

  setupSimulatorControls() {
    const carbRed = document.getElementById('inputSimCarbRed');
    const walkMins = document.getElementById('inputSimWalkMins');

    carbRed?.addEventListener('input', (e) => {
      this.simControls.carbReductionPct = parseFloat(e.target.value);
      document.getElementById('valSimCarbRed').textContent = `${e.target.value}%`;
      this.renderSimulation();
    });

    walkMins?.addEventListener('input', (e) => {
      this.simControls.walkMins = parseFloat(e.target.value);
      document.getElementById('valSimWalkMins').textContent = `${e.target.value} mins`;
      this.renderSimulation();
    });

    // Trajectory canvas touch & mouse scrubbing
    const canvas = document.getElementById('trajectoryCanvas');
    if (canvas) {
      const handleScrub = (clientX) => {
        const rect = canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const pct = Math.max(0, Math.min(1, x / rect.width));
        this.scrubberMinute = Math.round(pct * 120);
        this.drawTrajectoryCanvas();
      };

      canvas.addEventListener('mousemove', (e) => handleScrub(e.clientX));
      canvas.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0) handleScrub(e.touches[0].clientX);
      }, { passive: true });

      canvas.addEventListener('mouseleave', () => {
        this.scrubberMinute = null;
        this.drawTrajectoryCanvas();
      });
      canvas.addEventListener('touchend', () => {
        this.scrubberMinute = null;
        this.drawTrajectoryCanvas();
      });
    }
  }

  renderActiveProfile() {
    const p = this.activeProfile;
    if (!p) return;

    document.getElementById('dashGlucoseVal').textContent = Math.round(p.current_glucose);
    document.getElementById('dashHeartRate').innerHTML = `${p.heart_rate} <small style="font-size: 10px;">BPM</small>`;
    document.getElementById('dashMets').innerHTML = `${p.current_mets.toFixed(1)} <small style="font-size: 10px;">METs</small>`;
    document.getElementById('dashTIR').textContent = `${p.tir_daily}%`;
    document.getElementById('dashCohortBadge').textContent = p.group;
    document.getElementById('dashHbA1c').textContent = `${p.hba1c}%`;
    document.getElementById('dashFastingGlu').textContent = `${p.fasting_glu} mg/dL`;
    document.getElementById('dashInsulin').textContent = `${p.insulin} µIU/mL`;
    document.getElementById('dashBMI').textContent = `${p.bmi} kg/m²`;

    document.getElementById('dashRichness').textContent = `${p.microbial_richness} species`;
    document.getElementById('dashButyrate').textContent = `Optimal (Score ${p.butyrate_score}/5)`;
    document.getElementById('dashGutHealth').textContent = `${p.gut_health_score} / 100`;

    // Trend Pill color
    const pill = document.getElementById('dashTrendPill');
    if (p.current_glucose > 140) {
      pill.style.background = 'rgba(239, 68, 68, 0.15)';
      pill.style.color = '#ef4444';
      pill.style.borderColor = 'rgba(239, 68, 68, 0.3)';
      pill.innerHTML = `<span>↗</span> <span>Elevated</span>`;
    } else {
      pill.style.background = 'rgba(16, 185, 129, 0.15)';
      pill.style.color = '#10b981';
      pill.style.borderColor = 'rgba(16, 185, 129, 0.3)';
      pill.innerHTML = `<span>➔</span> <span>Steady</span>`;
    }
  }

  renderPresets() {
    const container = document.getElementById('presetChipsContainer');
    if (!container) return;
    container.innerHTML = '';

    const meals = this.activeProfile.sample_meals || [];
    meals.forEach((m, idx) => {
      const chip = document.createElement('div');
      chip.className = `preset-chip ${idx === 0 ? 'active' : ''}`;
      chip.textContent = m.name;
      chip.addEventListener('click', () => {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        this.selectPresetMeal(m);
      });
      container.appendChild(chip);
    });

    if (meals.length > 0) {
      this.selectPresetMeal(meals[0]);
    }
  }

  selectPresetMeal(meal) {
    this.currentMeal = { ...meal };
    document.getElementById('inputCarbs').value = meal.carbs;
    document.getElementById('inputFiber').value = meal.fiber;
    document.getElementById('inputProtein').value = meal.protein;
    document.getElementById('inputFat').value = meal.fat;

    document.getElementById('valCarbs').textContent = `${meal.carbs} g`;
    document.getElementById('valFiber').textContent = `${meal.fiber} g`;
    document.getElementById('valProtein').textContent = `${meal.protein} g`;
    document.getElementById('valFat').textContent = `${meal.fat} g`;
    document.getElementById('totalCaloriesVal').textContent = `${meal.calories} kcal`;
  }

  /* ========================================================================
     PHYSIOLOGICAL DIGITAL TWIN SIMULATION ENGINE
     Calibrated to TransformerForecaster & CounterfactualSimulator
     ======================================================================== */
  calculateTrajectory(g0, carbs, fiber, protein, fat, walkMins = 0, walkMets = 1.2) {
    const p = this.activeProfile;
    // Insulin resistance factor from HbA1c, fasting insulin, and VAE component
    const irFactor = Math.max(0.45, 1.45 - (p.hba1c - 5.0) * 0.22 - (p.insulin - 8.0) * 0.025);
    const netCarbs = Math.max(4.0, carbs - 0.5 * fiber);

    // Peak excursion rise formula
    const rawRise = (netCarbs * 0.95) / irFactor;
    // Protein and fat slightly delay and blunt the peak
    const fatBlunt = Math.max(0.75, 1.0 - (fat * 0.003));
    const maxDelta = rawRise * fatBlunt;

    // Generate 24 time steps (5-minute intervals up to 120 minutes)
    const curve = [];
    const timeSteps = [];
    for (let t = 5; t <= 120; t += 5) {
      timeSteps.push(t);
      // Bateman / Gamma absorption curve function
      const peakTime = 50.0; // Minutes to peak
      const riseFraction = (t / peakTime) * Math.exp(1.0 - (t / peakTime));
      let deltaG = maxDelta * riseFraction;

      // Skeletal muscle physical activity glucose uptake factor
      if (walkMins > 0 && t >= 20) {
        const walkEffective = Math.min(1.0, (walkMins / 30.0));
        const metsFactor = Math.max(1.0, walkMets / 3.0);
        const clearance = Math.min(0.45, 0.006 * walkMins * metsFactor);
        deltaG = deltaG * (1.0 - clearance);
      }

      const absoluteG = g0 + Math.max(0.0, deltaG);
      curve.push(absoluteG);
    }

    return { timeSteps, curve };
  }

  renderSimulation() {
    const g0 = this.activeProfile.current_glucose;
    const { carbs, fiber, protein, fat } = this.currentMeal;

    const summaryEl = document.getElementById('simMealSummary');
    if (summaryEl) {
      summaryEl.textContent = `${this.currentMeal.name} (${carbs}g Carbs) | Baseline: ${Math.round(g0)} mg/dL`;
    }

    // 1. Scenario 1: Baseline unmodified
    const s1 = this.calculateTrajectory(g0, carbs, fiber, protein, fat, 0, 1.2);

    // 2. Scenario 2: Dietary Carb reduction
    const redCarbs = carbs * (1.0 - this.simControls.carbReductionPct / 100.0);
    const s2 = this.calculateTrajectory(g0, redCarbs, fiber, protein, fat, 0, 1.2);

    // 3. Scenario 3: Post-Meal Walking
    const s3 = this.calculateTrajectory(g0, carbs, fiber, protein, fat, this.simControls.walkMins, this.simControls.walkMets);

    // 4. Scenario 4: Combined Intervention
    const s4 = this.calculateTrajectory(g0, redCarbs, fiber, protein, fat, this.simControls.walkMins, this.simControls.walkMets);

    this.simCurves = {
      s1: s1.curve,
      s2: s2.curve,
      s3: s3.curve,
      s4: s4.curve,
      timeSteps: s1.timeSteps
    };

    // Calculate metrics and rank scenarios
    const evaluateScenario = (name, curve, color) => {
      const peak = Math.max(...curve);
      const rise = peak - g0;
      const inRangeCount = curve.filter(v => v >= 70 && v <= 180).length;
      const tir = (inRangeCount / curve.length) * 100.0;

      // Clinical Utility Score U in [0, 1]
      const tScore = tir / 100.0;
      const pScore = Math.max(0.0, 1.0 - (peak - 70.0) / 180.0);
      const rScore = Math.max(0.0, 1.0 - rise / 100.0);
      const utility = (0.50 * tScore + 0.30 * pScore + 0.20 * rScore);

      return { name, curve, peak, rise, tir, utility, color };
    };

    const scenarios = [
      evaluateScenario("1. Baseline Meal", s1.curve, "#ef4444"),
      evaluateScenario(`2. -${this.simControls.carbReductionPct}% Carbohydrates`, s2.curve, "#f97316"),
      evaluateScenario(`3. +${this.simControls.walkMins}m Post-Meal Walk`, s3.curve, "#10b981"),
      evaluateScenario("4. Combined Intervention", s4.curve, "#3b82f6")
    ].sort((a, b) => b.utility - a.utility);

    this.renderRankedCards(scenarios);
    this.drawTrajectoryCanvas();
  }

  renderRankedCards(scenarios) {
    const container = document.getElementById('scenariosContainer');
    if (!container) return;
    container.innerHTML = '';

    scenarios.forEach((sc, idx) => {
      const card = document.createElement('div');
      card.className = `scenario-card ${idx === 0 ? 'top-rank' : ''}`;
      card.innerHTML = `
        <div class="scenario-header">
          <div class="scenario-name">
            <span style="display:inline-block; width:10px; height:10px; border-radius:2px; background:${sc.color};"></span>
            ${sc.name}
          </div>
          <span class="utility-badge">Score: ${(sc.utility * 100).toFixed(0)}</span>
        </div>
        <div class="scenario-metrics">
          <div class="metric-pill">
            Peak Glucose
            <span style="color: ${sc.peak > 180 ? '#ef4444' : '#10b981'};">${sc.peak.toFixed(0)} mg/dL</span>
          </div>
          <div class="metric-pill">
            Glucose Rise
            <span>+${sc.rise.toFixed(0)} mg/dL</span>
          </div>
          <div class="metric-pill">
            Time-In-Range
            <span style="color: #10b981;">${sc.tir.toFixed(0)}%</span>
          </div>
        </div>
      `;
      container.appendChild(card);
    });

    // Update AI recommendation
    const best = scenarios[0];
    const base = scenarios.find(s => s.name.includes("Baseline"));
    const drop = base ? (base.peak - best.peak).toFixed(0) : 25;

    const recText = document.getElementById('aiRecommendationText');
    if (recText) {
      recText.innerHTML = `
        <strong>Optimal Digital Twin Recommendation:</strong>
        ${best.name === "1. Baseline Meal" 
          ? "Your glycemic baseline handles this meal well within target range (<180 mg/dL)." 
          : `Applying <em>${best.name.replace(/^\d+\.\s*/, '')}</em> flattens your excursion by <strong>${drop} mg/dL</strong>, securing ${best.tir.toFixed(0)}% time-in-range.`}
      `;
    }
  }

  drawTrajectoryCanvas() {
    const canvas = document.getElementById('trajectoryCanvas');
    if (!canvas || !this.simCurves) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 32;
    const padRight = 14;
    const padTop = 15;
    const padBottom = 22;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const minY = 60;
    const maxY = 220;

    const mapX = (t) => padLeft + (t / 120.0) * plotW;
    const mapY = (val) => padTop + plotH - ((val - minY) / (maxY - minY)) * plotH;

    // 1. Draw Target Band (70 - 180 mg/dL)
    const y180 = mapY(180);
    const y70 = mapY(70);
    const targetGrad = ctx.createLinearGradient(0, y180, 0, y70);
    targetGrad.addColorStop(0, 'rgba(5, 150, 105, 0.12)');
    targetGrad.addColorStop(1, 'rgba(5, 150, 105, 0.03)');
    ctx.fillStyle = targetGrad;
    ctx.fillRect(padLeft, y180, plotW, y70 - y180);

    // Target boundary dotted lines
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(5, 150, 105, 0.45)';
    ctx.lineWidth = 1;
    [180, 70].forEach(val => {
      ctx.beginPath();
      ctx.moveTo(padLeft, mapY(val));
      ctx.lineTo(w - padRight, mapY(val));
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Y-Axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Outfit, sans-serif';
    ctx.textAlign = 'right';
    [70, 120, 180].forEach(val => {
      ctx.fillText(String(val), padLeft - 6, mapY(val) + 3);
    });

    // 2. Plot Curves
    const curvesToPlot = [
      { data: this.simCurves.s1, color: '#dc2626', lw: 2.2 },
      { data: this.simCurves.s2, color: '#ea580c', lw: 2.2 },
      { data: this.simCurves.s3, color: '#059669', lw: 2.6 },
      { data: this.simCurves.s4, color: '#2563eb', lw: 3.2 }
    ];

    curvesToPlot.forEach(cp => {
      ctx.beginPath();
      ctx.strokeStyle = cp.color;
      ctx.lineWidth = cp.lw;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      this.simCurves.timeSteps.forEach((t, i) => {
        const x = mapX(t);
        const y = mapY(cp.data[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // 3. Draw Interactive Scrubber if active
    if (this.scrubberMinute !== null) {
      const scrubX = mapX(this.scrubberMinute);
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.4)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 2]);
      ctx.moveTo(scrubX, padTop);
      ctx.lineTo(scrubX, padTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      const idx = Math.min(this.simCurves.timeSteps.length - 1, Math.max(0, Math.round(this.scrubberMinute / 5) - 1));
      const val = this.simCurves.s1[idx];

      const tooltip = document.getElementById('scrubberTooltip');
      if (tooltip) {
        tooltip.style.display = 'block';
        tooltip.innerHTML = `<strong>t: ${this.scrubberMinute}m</strong> | Baseline: ${val.toFixed(0)} mg/dL`;
      }
    } else {
      const tooltip = document.getElementById('scrubberTooltip');
      if (tooltip) tooltip.style.display = 'none';
    }
  }

  /* ========================================================================
     GUT MICROBIOME VAE RADAR CANVAS
     ======================================================================== */
  renderMicrobiomeRadar() {
    const canvas = document.getElementById('microbiomeRadarCanvas');
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 25;

    const latents = this.activeProfile.microbiome_vae || [0.2, -0.3, 0.5, 0.1, -0.2, 0.4, -0.1, 0.3];
    const numAxes = 8;

    // Normalize latents from [-1.5, 1.5] into [0.2, 1.0] for radar radius
    const normValues = latents.map(v => Math.max(0.15, Math.min(1.0, 0.5 + (v / 2.0))));

    // Draw concentric radar web
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1.2;
    for (let rPct = 0.25; rPct <= 1.0; rPct += 0.25) {
      ctx.beginPath();
      for (let i = 0; i < numAxes; i++) {
        const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
        const x = cx + radius * rPct * Math.cos(angle);
        const y = cy + radius * rPct * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw axis lines and labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Outfit, sans-serif';
    ctx.textAlign = 'center';

    for (let i = 0; i < numAxes; i++) {
      const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
      const x = cx + radius * Math.cos(angle);
      const y = cy + radius * Math.sin(angle);

      ctx.beginPath();
      ctx.strokeStyle = '#e2e8f0';
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();

      const labelX = cx + (radius + 14) * Math.cos(angle);
      const labelY = cy + (radius + 14) * Math.sin(angle) + 4;
      ctx.fillText(`VAE${i + 1}`, labelX, labelY);
    }

    // Draw filled VAE polygon
    ctx.beginPath();
    normValues.forEach((val, i) => {
      const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
      const x = cx + radius * val * Math.cos(angle);
      const y = cy + radius * val * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    ctx.fillStyle = 'rgba(124, 58, 237, 0.16)';
    ctx.fill();
    ctx.strokeStyle = '#7c3aed';
    ctx.lineWidth = 2.4;
    ctx.stroke();

    // Data points
    normValues.forEach((val, i) => {
      const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
      const x = cx + radius * val * Math.cos(angle);
      const y = cy + radius * val * Math.sin(angle);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#7c3aed';
      ctx.fill();
    });
  }

  /* ========================================================================
     DASHBOARD MINI CGM SPARKLINE
     ======================================================================== */
  drawSparkline() {
    const canvas = document.getElementById('cgmSparklineCanvas');
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    // Mock recent 24 readings leading to current glucose
    const curr = this.activeProfile.current_glucose;
    const points = [];
    for (let i = 0; i < 24; i++) {
      const noise = (Math.sin(i * 0.5) * 12) + (Math.cos(i * 0.3) * 6);
      points.push(curr - 18 + noise + (i * 0.8));
    }
    points[points.length - 1] = curr;

    const minY = 70;
    const maxY = 180;
    const mapY = (v) => h - ((v - minY) / (maxY - minY)) * h;
    const stepX = w / (points.length - 1);

    // Fill gradient
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = i * stepX;
      const y = Math.max(4, Math.min(h - 4, mapY(p)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();

    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(2, 132, 199, 0.18)');
    grad.addColorStop(1, 'rgba(2, 132, 199, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = i * stepX;
      const y = Math.max(4, Math.min(h - 4, mapY(p)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Current point dot
    const lastX = w - 1;
    const lastY = mapY(curr);
    ctx.beginPath();
    ctx.arc(lastX - 2, lastY, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// Instantiate on DOM content loaded
document.addEventListener('DOMContentLoaded', () => {
  window.glycoApp = new GlycoTwinApp();
});
