// static/js/schedule.js

document.addEventListener("DOMContentLoaded", () => {
    // 1. DOM Elements (Safe Selection)
    const getEl = (id) => document.getElementById(id);

    const els = {
        monthInput: getEl("month-input"),
        prevMonthBtn: getEl("prev-month-btn"),
        nextMonthBtn: getEl("next-month-btn"),
        currMonthBtn: getEl("curr-month-btn"),

        generateBtn: getEl("generate-btn"),
        stopBtn: getEl("stop-btn"),
        downloadBtn: getEl("download-schedule-btn"),

        tableWrapper: getEl("schedule-table-wrapper"),
        metaDiv: getEl("schedule-meta"),
        titleDiv: getEl("schedule-month-title"),

        optModeRadios: document.querySelectorAll('input[name="opt_mode"]'),
        manualOptionsDiv: getEl("manual-options"),
        // manualDetailSelect Removed

        trainMlBtn: getEl("train-ml-btn"),
        trainDeepBtn: getEl("train-deep-btn"),

        nCandidatesInput: getEl("n_candidates"),
        topKInput: getEl("top_k"),
        lsIterationsInput: getEl("ls_iterations"),
        teamCountInput: getEl("team-count-input"),

        dashContainer: getEl("dashboard-container"),
        dashScore: getEl("dash-score"),
        dashComment: getEl("dash-comment"),
        dashViolations: getEl("dash-violations"),

        loadingOverlay: getEl("loading-overlay"),
        loadingStep: getEl("loading-step"),
        loadingDesc: getEl("loading-desc"),

        cellEditor: getEl("cell-editor"),
        cellShiftSelect: getEl("cell-shift-select"),
        cellLockedCheckbox: getEl("cell-locked-checkbox"),

        configToggle: getEl("config-toggle"),
        configBody: getEl("config-body")
    };

    let currentController = null;
    let currentScheduleData = null;
    let currentScheduleId = null;
    let nurses = [];
    let nurseMap = {};
    const STORAGE_KEY = "nsp_schedule_opts";

    // 2. Init
    async function init() {
        // Restore Options
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        if(saved.optMode) {
            const r = document.querySelector(`input[name="opt_mode"][value="${saved.optMode}"]`);
            if(r) r.checked = true;
        }
        toggleManualOptions();

        // Set Month
        setMonthInput(new Date());

        // Load Data
        await loadNurses();
        if(window.ScoreRulesPanel && window.appConfig) {
             ScoreRulesPanel.init(window.appConfig.wardId);
        }

        // Events
        if(els.prevMonthBtn) els.prevMonthBtn.onclick = () => changeMonth(-1);
        if(els.nextMonthBtn) els.nextMonthBtn.onclick = () => changeMonth(1);
        if(els.currMonthBtn) els.currMonthBtn.onclick = () => setMonthInput(new Date());
        if(els.monthInput) els.monthInput.onchange = loadSchedule;

        if(els.generateBtn) els.generateBtn.onclick = generateSchedule;
        if(els.stopBtn) els.stopBtn.onclick = stopGeneration;
        if(els.downloadBtn) els.downloadBtn.onclick = () => {
            if(currentScheduleId) location.href = `/api/schedules/${currentScheduleId}/download_excel`;
        };

        els.optModeRadios.forEach(r => r.onchange = () => { toggleManualOptions(); saveOptions(); });

        if(els.trainMlBtn) els.trainMlBtn.onclick = () => runTrain("ml");
        if(els.trainDeepBtn) els.trainDeepBtn.onclick = () => runTrain("deep");

        if(els.configToggle && els.configBody) {
            els.configToggle.onclick = () => {
                const isHidden = els.configBody.classList.toggle("hidden");
                const icon = els.configToggle.querySelector(".toggle-icon");
                if(icon) icon.textContent = isHidden ? "▼" : "▲";
            };
        }

        // Cell Editor
        if(els.cellShiftSelect) els.cellShiftSelect.onchange = onCellShiftChange;
        if(els.cellLockedCheckbox) els.cellLockedCheckbox.onchange = onCellLockedChange;
        document.addEventListener("click", onDocumentClick);

        loadSchedule();
    }

    // Helpers
    function setMonthInput(date) {
        if(els.monthInput) {
            els.monthInput.value = `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}`;
            loadSchedule();
        }
    }
    function changeMonth(delta) {
        if(!els.monthInput) return;
        const [y, m] = els.monthInput.value.split("-").map(Number);
        setMonthInput(new Date(y, m-1+delta, 1));
    }
    function saveOptions() {
        const mode = document.querySelector('input[name="opt_mode"]:checked')?.value;
        if(mode) localStorage.setItem(STORAGE_KEY, JSON.stringify({ optMode: mode }));
    }
    function toggleManualOptions() {
        const mode = document.querySelector('input[name="opt_mode"]:checked')?.value;
        if (mode === "manual_default") {
            if(els.manualOptionsDiv) els.manualOptionsDiv.classList.remove("hidden");
            if(els.trainMlBtn) els.trainMlBtn.classList.add("hidden");
            if(els.trainDeepBtn) els.trainDeepBtn.classList.add("hidden");
        } else {
            if(els.manualOptionsDiv) els.manualOptionsDiv.classList.add("hidden");
            if(mode === "ml") {
                if(els.trainMlBtn) els.trainMlBtn.classList.remove("hidden");
                if(els.trainDeepBtn) els.trainDeepBtn.classList.add("hidden");
            } else {
                if(els.trainMlBtn) els.trainMlBtn.classList.add("hidden");
                if(els.trainDeepBtn) els.trainDeepBtn.classList.remove("hidden");
            }
        }
    }

    // Load Data
    async function loadNurses() {
        try {
            const res = await fetch(`/api/nurses?ward_id=${window.appConfig.wardId}`);
            nurses = await res.json();
            nurses.forEach(n => nurseMap[n.id] = n);
            // Team Count
            const teams = new Set();
            nurses.forEach(n => { if(n.team) teams.add(n.team); });
            const cnt = teams.size > 0 ? teams.size : 3;
            if(els.teamCountInput) {
                els.teamCountInput.value = cnt;
                updateShiftNeeds(cnt);
            }
        } catch(e) { console.error("Load Nurse Error", e); }
    }

    async function loadSchedule() {
        if(!els.monthInput) return;
        const month = els.monthInput.value;
        if(!month) return;

        if(els.tableWrapper) els.tableWrapper.innerHTML = '<div class="loading-placeholder">데이터 로드 중...</div>';
        if(els.dashContainer) els.dashContainer.classList.add("hidden");
        if(els.titleDiv) els.titleDiv.textContent = `📅 스케줄표 (${month})`;

        try {
            const res = await fetch(`/api/schedules/by_month?month=${month}`);
            const data = await res.json();

            if(data.exists) {
                currentScheduleId = data.schedule_id;
                currentScheduleData = data.schedule;
                renderTable(data.schedule);
                renderMeta(data.schedule);

                let breakdown = data.schedule.score_breakdown || {};
                if(typeof breakdown === 'string') { try { breakdown = JSON.parse(breakdown); } catch(e){} }
                updateDashboard(data.schedule.score_total, breakdown, "기존 스케줄 로드됨");
            } else {
                await createEmptySchedule(month);
            }
        } catch(e) {
            console.error(e);
            if(els.tableWrapper) els.tableWrapper.innerHTML = '<div class="empty-state">로드 오류</div>';
        }
    }

    async function createEmptySchedule(month) {
        try {
            const res = await fetch("/api/schedules/create_empty", {
                method: "POST", headers: {"Content-Type":"application/json"},
                body: JSON.stringify({ month: month, created_by: "auto_load" })
            });
            const data = await res.json();
            currentScheduleId = data.schedule_id;
            currentScheduleData = data.schedule;
            renderTable(data.schedule);
            renderMeta(data.schedule);
            updateDashboard(0, null, "빈 스케줄 생성됨 (입력 가능)");
        } catch(e) {
            if(els.tableWrapper) els.tableWrapper.innerHTML = '<div class="empty-state">스케줄 생성 실패</div>';
        }
    }

    // === Generation Logic ===
    async function generateSchedule() {
        if(!els.monthInput) return;
        const month = els.monthInput.value;
        if(!currentScheduleId) return alert("스케줄이 로드되지 않았습니다.");
        if(!confirm("스케줄을 생성하시겠습니까? (기존 내용은 덮어씌워집니다)")) return;

        currentController = new AbortController();
        setGeneratingState(true);

        const mode = document.querySelector('input[name="opt_mode"]:checked')?.value || "manual_default";

        // [Fix] AI 모드일 때만 학습 진행
        if (mode !== "manual_default") {
            updateLoadingStep(1, 4, "AI 학습 및 데이터 준비 중...");
            try {
                if (mode === "ml") {
                    await fetch("/api/ml/train", { method: "POST" });
                } else if (mode === "pytorch") {
                    await fetch("/api/deep/train", { method: "POST" });
                }
            } catch(e) { console.warn("Training skipped/failed", e); }
        } else {
            updateLoadingStep(1, 4, "설정 확인 중...");
            await new Promise(r => setTimeout(r, 500));
        }

        try {
            // Step 2: Save Config
            updateLoadingStep(2, 4, "설정 저장 및 초기화...");
            const shiftNeeds = collectShiftNeeds();
            // ScoreRulesPanel Check
            const scoreRules = window.ScoreRulesPanel ? ScoreRulesPanel.getConfigJson() : {};

            await fetch(`/api/schedules/${currentScheduleId}/save`, {
                method: "POST", headers: {"Content-Type":"application/json"},
                body: JSON.stringify({
                    schedule: {
                        grid: currentScheduleData.grid, locks: currentScheduleData.locks,
                        config: { ...scoreRules, shift_needs: shiftNeeds }
                    },
                    changed_by: "system", record_history: false
                }),
                signal: currentController.signal
            });

            // Step 3: Generate
            startProgressSimulation();
            const genMode = document.querySelector('input[name="gen_mode"]:checked')?.value || "heuristic";
            const lsMode = document.querySelector('input[name="ls_mode"]:checked')?.value || "random";
            const surMode = document.querySelector('input[name="surrogate_mode"]:checked')?.value || "none";

            let nCand = null, topK = null, iter = null;

            if (mode === "manual_default") {
                if(els.nCandidatesInput?.value) nCand = parseInt(els.nCandidatesInput.value);
                if(els.topKInput?.value) topK = parseInt(els.topKInput.value);
                if(els.lsIterationsInput?.value) iter = parseInt(els.lsIterationsInput.value);
            }

            const planRes = await fetch(`/api/schedules/by_month?month=${month}`);
            const planData = await planRes.json();
            const planId = planData.plan_id;

            const payload = {
                plan_id: planId,
                ward_id: window.appConfig.wardId,
                optimization_mode: mode,
                generator_mode: genMode,
                local_search_mode: lsMode,
                surrogate_mode: surMode,
                n_candidates: nCand, top_k: topK, iterations: iter,
                seed: Math.floor(Math.random() * 9999)
            };

            const genRes = await fetch("/api/schedules/generate", {
                method: "POST", headers: {"Content-Type":"application/json"},
                body: JSON.stringify(payload),
                signal: currentController.signal
            });

            if(!genRes.ok) throw new Error("Generation Failed");
            const genData = await genRes.json();

            // Step 4: Finish
            updateLoadingStep(4, 4, "완료!");
            currentScheduleId = genData.schedule_id;
            await loadSchedule();

            let breakdown = genData.score_breakdown || {};
            if(typeof breakdown === 'string') { try { breakdown = JSON.parse(breakdown); } catch(e){} }
            updateDashboard(genData.score, breakdown, "AI 생성 완료");

        } catch(e) {
            if(e.name !== "AbortError") alert("오류: " + e.message);
        } finally {
            setGeneratingState(false);
            stopProgressSimulation();
        }
    }

    function stopGeneration() {
        if(currentController) {
            currentController.abort();
            currentController = null;
        }
    }

    function setGeneratingState(isGen) {
        if(isGen) {
            if(els.loadingOverlay) els.loadingOverlay.classList.remove("hidden");
            if(els.generateBtn) els.generateBtn.style.display = "none";
            if(els.stopBtn) {
                els.stopBtn.classList.remove("hidden");
                els.stopBtn.style.display = "inline-block";
            }
            document.querySelectorAll("input, select, button:not(#stop-btn)").forEach(el => el.disabled = true);
        } else {
            if(els.loadingOverlay) els.loadingOverlay.classList.add("hidden");
            if(els.generateBtn) {
                els.generateBtn.classList.remove("hidden");
                els.generateBtn.style.display = "inline-block";
            }
            if(els.stopBtn) {
                els.stopBtn.classList.add("hidden");
                els.stopBtn.style.display = "none";
            }
            document.querySelectorAll("input, select, button").forEach(el => el.disabled = false);
            if(els.teamCountInput) els.teamCountInput.readOnly = true;
        }
    }

    let progressTimer = null;
    function updateLoadingStep(step, total, msg) {
        if(els.loadingStep) els.loadingStep.textContent = `[${step}/${total}] 진행 중...`;
        if(els.loadingDesc) els.loadingDesc.textContent = msg;
    }
    function startProgressSimulation() {
        updateLoadingStep(3, 4, "최적의 스케줄 탐색 중...");
        let step = 0;
        const msgs = ["초기 후보군 생성...", "Hard Rule 검증...", "AI 최적화 수행...", "정밀 튜닝 중..."];
        if(progressTimer) clearInterval(progressTimer);
        progressTimer = setInterval(() => {
            if(els.loadingDesc) els.loadingDesc.textContent = msgs[step % msgs.length];
            step++;
        }, 1500);
    }
    function stopProgressSimulation() { if(progressTimer) clearInterval(progressTimer); }

    function updateDashboard(score, breakdown, msg) {
        if(els.dashContainer) els.dashContainer.classList.remove("hidden");
        if(els.dashScore) {
            els.dashScore.textContent = score ? score.toFixed(1) : "0.0";
            els.dashScore.style.color = score >= 0 ? "#166534" : "#b91c1c";
        }
        if(els.dashComment) els.dashComment.textContent = msg;
        if(els.dashViolations) {
            els.dashViolations.innerHTML = "";
            if (breakdown && breakdown.by_rule && Array.isArray(breakdown.by_rule)) {
                const bads = breakdown.by_rule.filter(r => r.score < 0).sort((a,b) => a.score - b.score);
                if(bads.length === 0) {
                    els.dashViolations.innerHTML = "<li class='good'>✅ 감점 사항 없음</li>";
                } else {
                    bads.forEach(v => {
                        const li = document.createElement("li");
                        const name = mapRuleName(v.rule_id);
                        let detailText = "";
                        if(v.details) {
                            if(v.details.desc) detailText = ` - ${v.details.desc}`;
                            else if(v.details.pattern) detailText = ` (패턴: ${v.details.pattern})`;
                        }
                        li.innerHTML = `<strong>${name}</strong>: ${v.score.toFixed(1)}점 ${detailText}`;
                        if(v.score < -500) {
                             li.style.fontWeight = "bold"; li.style.color = "#991b1b";
                        }
                        els.dashViolations.appendChild(li);
                    });
                }
            } else els.dashViolations.innerHTML = "<li>(상세 내역 없음)</li>";
        }
    }

    function mapRuleName(id) {
        const map = {
            "CONSEC_WORK_5_PEN": "5일 연속 근무",
            "WORK_O_WORK_PEN": "퐁당퐁당(W-O-W)", // [Fix] Text Update
            "N_COUNT_MAX_16": "월 나이트 과다",
            "FAIRNESS_MIN_SET": "공정성 불균형",
            "INVALID_SHIFT_CODE": "잘못된 근무코드",
            "UM_LOCK_VIOLATION": "고정 근무 위반",
            "TEAM_COVERAGE_FAIL": "팀 최소인원 미달",
            "LEADER_REQUIRED": "리더 부재",
            "SPECIAL_NO_NIGHT": "특수(임산부 등) N금지 위반",
            "TAG_PREGNANT_N_BAN": "임산부 야간근무",
            "TAG_N_KEEP_BAN": "N전담 타근무"
        };
        if(id.startsWith("HARD_")) return "🚨 " + (map[id.replace("HARD_", "")] || id);
        return map[id] || id;
    }

    // === Table Render (Fixed) ===
    function renderTable(schedule) {
        const grid = schedule.grid || {};
        const locks = schedule.locks || {};
        let dates = Object.keys(grid).sort();

        // [Fix] Grid가 비어있으면(빈 스케줄), 선택된 달의 모든 날짜를 강제로 생성
        if (dates.length === 0 && els.monthInput.value) {
             const [y, m] = els.monthInput.value.split("-").map(Number);
             const daysInMonth = new Date(y, m, 0).getDate();
             for(let i=1; i<=daysInMonth; i++) {
                 dates.push(`${y}-${String(m).padStart(2, '0')}-${String(i).padStart(2, '0')}`);
             }
        }

        if (dates.length === 0) {
             if(els.tableWrapper) els.tableWrapper.innerHTML = "<div class='empty-state'>표시할 날짜가 없습니다.</div>";
             return;
        }

        // 2. 리더(자격 보유자 중 최고 연차) 계산
        const leadersMap = {};
        dates.forEach(d => {
            leadersMap[d] = {};
            const dayData = grid[d] || {};
            ['D', 'E', 'N'].forEach(shiftType => {
                const assignedIds = dayData[shiftType] || [];
                let workers = [];
                if(Array.isArray(assignedIds)) {
                     assignedIds.forEach(id => {
                         if(nurseMap[id]) workers.push(nurseMap[id]);
                     });
                }
                const propKey = `leader_${shiftType.toLowerCase()}`;
                const qualifiedWorkers = workers.filter(n => n[propKey] === true);

                if (qualifiedWorkers.length > 0) {
                    qualifiedWorkers.sort((a, b) => (b.years_exp||0) - (a.years_exp||0));
                    leadersMap[d][shiftType] = qualifiedWorkers[0].id;
                }
            });
        });

        const sortedNurses = [...nurses].sort((a, b) => {
            const ta = (a.team || "").trim(), tb = (b.team || "").trim();
            if (ta === tb) return a.name.localeCompare(b.name);
            if (ta === "") return -1; if (tb === "") return 1;
            return ta.localeCompare(tb);
        });

        let html = '<table class="schedule-table"><thead><tr><th class="col-name">이름</th><th class="col-team">팀</th>';
        dates.forEach(d => {
            const day = parseInt(d.split("-")[2]);
            html += `<th class="col-day">${day}</th>`;
        });
        html += '<th class="col-stats">D/E/N/O</th></tr></thead><tbody>';

        let lastTeam = null;
        sortedNurses.forEach((n, idx) => {
            const currentTeam = (n.team || "").trim();
            if (idx > 0 && currentTeam !== lastTeam) {
                html += `<tr class="team-separator-row"><td colspan="${dates.length + 3}"></td></tr>`;
            }
            lastTeam = currentTeam;

            let cD=0, cE=0, cN=0, cO=0;
            const teamLabel = currentTeam || "-";

            html += `<tr><td class="col-name">${n.name}</td><td class="col-team">${teamLabel}</td>`;

            dates.forEach(d => {
                let shift = "";
                const dayData = grid[d] || {};
                if(dayData && typeof dayData === 'object') {
                    for(const [s, ids] of Object.entries(dayData)) {
                        if(Array.isArray(ids) && ids.includes(n.id)) { shift = s; break; }
                    }
                }

                if(shift==="D"||shift==="C") cD++;
                else if(shift==="E"||shift==="M") cE++;
                else if(shift==="N") cN++;
                else if(shift==="O"||shift==="P") cO++;

                const isLocked = locks[d] && locks[d][String(n.id)];
                const lockClass = isLocked ? "locked-cell" : "";

                let leaderBadge = "";
                if (['D', 'E', 'N'].includes(shift)) {
                    if (leadersMap[d] && leadersMap[d][shift] === n.id) {
                        leaderBadge = `<span class="leader-badge">★</span>`;
                    }
                }

                html += `<td class="day-cell shift-${shift} ${lockClass}"
                             data-date="${d}" data-nid="${n.id}" data-shift="${shift}">
                             ${shift}${isLocked?'🔒':''}${leaderBadge}
                         </td>`;
            });
            html += `<td class="col-stats">${cD}/${cE}/${cN}/${cO}</td></tr>`;
        });
        html += '</tbody></table>';
        if(els.tableWrapper) els.tableWrapper.innerHTML = html;
        document.querySelectorAll(".day-cell").forEach(td => td.addEventListener("click", (e) => openCellEditor(e.target)));
    }

    function renderMeta(schedule) {
        if(els.metaDiv && schedule.generated_at) {
            const dt = new Date(schedule.generated_at);
            els.metaDiv.textContent = `생성: ${dt.toLocaleString()} (ID: ${schedule.id})`;
        } else if(els.metaDiv) {
            els.metaDiv.textContent = "";
        }
    }

    // Cell Editor
    function openCellEditor(td) {
        if(!td.classList.contains("day-cell")) return;
        currentEditingCell = td;
        const rect = td.getBoundingClientRect();

        const top = window.scrollY + rect.bottom + 5;
        let left = rect.left;
        if (left + 220 > window.innerWidth) left = window.innerWidth - 230;

        if(els.cellEditor) {
            els.cellEditor.style.top = top + "px";
            els.cellEditor.style.left = left + "px";
            els.cellEditor.classList.remove("hidden");
        }

        const nid = td.dataset.nid;
        const date = td.dataset.date;
        const shift = td.dataset.shift;
        const nurse = nurseMap[nid];

        if(nurse) document.getElementById("cell-info").textContent = `${nurse.name} (${date})`;
        if(els.cellLockedCheckbox) els.cellLockedCheckbox.checked = td.classList.contains("locked-cell");

        // [Fix] 7 Shifts + X buttons
        const shifts = ["D", "E", "N", "M", "C", "O", "P", "X"];
        const cont = document.getElementById("shift-options");
        if(cont) {
            cont.innerHTML = "";
            shifts.forEach(s => {
                const btn = document.createElement("button");
                btn.textContent = s === "X" ? "지우기" : s;
                btn.className = `shift-btn shift-${s}`;
                btn.onclick = () => updateCell(nid, date, s, els.cellLockedCheckbox.checked);
                cont.appendChild(btn);
            });
        }
    }

    function onCellShiftChange() {
        if(!currentEditingCell) return;
        const nid = currentEditingCell.dataset.nid;
        const date = currentEditingCell.dataset.date;
        const val = els.cellShiftSelect.value;
        const locked = els.cellLockedCheckbox.checked;
        updateCell(nid, date, val, locked);
    }

    function onCellLockedChange() {
        if(!currentEditingCell) return;
        const nid = currentEditingCell.dataset.nid;
        const date = currentEditingCell.dataset.date;
        const shift = currentEditingCell.dataset.shift;
        updateCell(nid, date, shift, els.cellLockedCheckbox.checked);
    }

    function onDocumentClick(e) {
        if(els.cellEditor && els.cellEditor.contains(e.target)) return;
        if(currentEditingCell && currentEditingCell.contains(e.target)) return;
        if(els.cellEditor) els.cellEditor.classList.add("hidden");
    }

    async function updateCell(nid, date, shift, locked) {
        if(!currentScheduleData.grid[date]) currentScheduleData.grid[date] = {};
        const dayMap = currentScheduleData.grid[date];

        for(const s in dayMap) if(Array.isArray(dayMap[s])) dayMap[s] = dayMap[s].filter(id => id !== parseInt(nid));
        if(shift !== "X") {
             if(!dayMap[shift]) dayMap[shift] = [];
             dayMap[shift].push(parseInt(nid));
        }

        if(!currentScheduleData.locks) currentScheduleData.locks = {};
        if(!currentScheduleData.locks[date]) currentScheduleData.locks[date] = {};
        currentScheduleData.locks[date][String(nid)] = locked;

        renderTable(currentScheduleData);

        try {
            const res = await fetch(`/api/schedules/${currentScheduleId}/save`, {
                method: "POST", headers: {"Content-Type":"application/json"},
                body: JSON.stringify({
                    schedule: currentScheduleData,
                    changed_by: "manual_edit", record_history: false
                })
            });
            const data = await res.json();

            // [Fix] Alert Violations
            if(data.violations && data.violations.length > 0) {
                 const msg = "⚠️ 위반 경고:\n" + data.violations.slice(0, 3).join("\n") + (data.violations.length > 3 ? "\n..." : "");
                 alert(msg);
                 if(data.score_breakdown) updateDashboard(data.score_total, data.score_breakdown, "수정 중 위반 발생");
            } else {
                 if(data.score_breakdown) updateDashboard(data.score_total, data.score_breakdown, "수정 저장됨");
            }
        } catch(e) { console.error("Save failed", e); }

        if(els.cellEditor) els.cellEditor.classList.add("hidden");
    }

    function updateShiftNeeds(teamCount) {
        const base = Math.max(1, parseInt(teamCount));
        document.querySelectorAll("#shift-needs-table input").forEach(inp => {
            const id = inp.id;
            if(id.endsWith("-n")) inp.value = base;
            else inp.value = base + 1;
        });
    }

    function collectShiftNeeds() {
        const needs = { weekday: {}, weekend: {}, holiday: {} };
        document.querySelectorAll("#shift-needs-table tr[data-type]").forEach(r => {
            const t = r.dataset.type;
            const inps = r.querySelectorAll("input");
            if(inps.length >= 3) {
                needs[t] = { D: parseInt(inps[0].value)||0, E: parseInt(inps[1].value)||0, N: parseInt(inps[2].value)||0 };
            }
        });
        return needs;
    }

    async function runTrain(type) {
        const btn = type === 'ml' ? els.trainMlBtn : els.trainDeepBtn;
        if(btn) {
            btn.disabled = true;
            btn.textContent = "학습 중...";
        }
        try {
            const res = await fetch(type === 'ml' ? "/api/ml/train" : "/api/deep/train", { method: "POST" });
            const d = await res.json();
            alert(d.message);
        } catch(e) { alert("학습 실패"); }
        finally {
            if(btn) {
                btn.disabled = false;
                btn.textContent = (type==='ml' ? "ML 학습" : "Deep 학습");
            }
        }
    }

    init();
});