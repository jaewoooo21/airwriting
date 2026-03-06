// ══════════════════════════════════════════════════════════════
// 3D Hand Mesh Widget — Realistic Human Hand Visualizer
// Renders a human-like hand with palm, 5 fingers, forearm segment,
// and a held pen, driven by IMU quaternion data (S1q/S2q/S3q).
// ══════════════════════════════════════════════════════════════
(function () {
    'use strict';

    const WIDGET_WIDTH = 280;
    const WIDGET_HEIGHT = 210;
    const LERP_FACTOR = 0.25;

    // ── Scene ──
    const canvas = document.getElementById('handWidgetCanvas');
    if (!canvas) return;

    canvas.width = WIDGET_WIDTH * Math.min(window.devicePixelRatio, 2);
    canvas.height = WIDGET_HEIGHT * Math.min(window.devicePixelRatio, 2);
    canvas.style.width = WIDGET_WIDTH + 'px';
    canvas.style.height = WIDGET_HEIGHT + 'px';

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(WIDGET_WIDTH, WIDGET_HEIGHT);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);

    const scene = new THREE.Scene();

    // ── Camera ──
    const cam = new THREE.PerspectiveCamera(38, WIDGET_WIDTH / WIDGET_HEIGHT, 0.01, 10);
    cam.position.set(0.15, 0.45, 0.55);
    cam.lookAt(0, 0.22, 0);

    // ── Lights ──
    scene.add(new THREE.AmbientLight(0x8899bb, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0.5, 1.0, 0.8);
    scene.add(dirLight);
    const rimLight = new THREE.DirectionalLight(0x4ade80, 0.25);
    rimLight.position.set(-0.5, 0.2, -0.5);
    scene.add(rimLight);

    // ── Subtle grid floor ──
    const gridHelper = new THREE.GridHelper(0.6, 10, 0x1a1a2e, 0x1a1a2e);
    gridHelper.position.y = -0.02;
    gridHelper.material.opacity = 0.3;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);

    // ── Color palette ──
    const COL_BONE = 0xcbd5e1;   // light gray for bones
    const COL_JOINT = 0x64748b;  // slate for joints
    const COL_PALM = 0x94a3b8;   // palm color
    const COL_FOREARM = 0x475569;
    const COL_PEN_IDLE = 0x38bdf8;
    const COL_PEN_ACTIVE = 0xff1a66;
    const COL_FINGER_HOLD = 0x4ade80; // index/middle (holding pen)
    const COL_FINGER_FREE = 0x94a3b8; // ring/pinky/thumb

    // ── Helper: create a bone segment (cylinder + joint sphere) ──
    function createBone(length, radius, color, emissive) {
        const group = new THREE.Group();

        // Cylinder
        const geom = new THREE.CylinderGeometry(radius, radius * 0.85, length, 6, 1);
        geom.translate(0, length / 2, 0);
        const solidMat = new THREE.MeshPhongMaterial({
            color: color, transparent: true, opacity: 0.45, shininess: 60
        });
        group.add(new THREE.Mesh(geom, solidMat));

        // Wireframe
        const wireMat = new THREE.MeshBasicMaterial({
            color: color, wireframe: true, transparent: true, opacity: 0.5
        });
        group.add(new THREE.Mesh(geom, wireMat));

        // Joint sphere at base
        const sphereGeom = new THREE.SphereGeometry(radius * 1.4, 8, 6);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: emissive || color,
            emissive: emissive || color,
            emissiveIntensity: 0.25,
            transparent: true, opacity: 0.5
        });
        group.add(new THREE.Mesh(sphereGeom, sphereMat));

        return group;
    }

    // ── Helper: build a finger chain ──
    function buildFinger(segLengths, radius, color, emissive) {
        const root = new THREE.Group();
        let parent = root;
        const joints = [];

        for (let i = 0; i < segLengths.length; i++) {
            const bone = createBone(segLengths[i], radius, color, emissive);
            parent.add(bone);
            joints.push(bone);

            const anchor = new THREE.Group();
            anchor.position.set(0, segLengths[i], 0);
            bone.add(anchor);
            parent = anchor;
        }

        return { root, joints, tipAnchor: parent };
    }

    // ══════════════════════════════════════════
    // Build the hand hierarchy
    // ══════════════════════════════════════════
    const armGroup = new THREE.Group();
    scene.add(armGroup);

    // S1: Forearm bone
    const forearmBone = createBone(0.22, 0.022, COL_FOREARM, 0x334155);
    armGroup.add(forearmBone);
    const forearmEnd = new THREE.Group();
    forearmEnd.position.set(0, 0.22, 0);
    forearmBone.add(forearmEnd);

    // S2: Wrist joint + Palm
    const wristGroup = new THREE.Group();
    forearmEnd.add(wristGroup);

    // Palm (flattened box)
    const palmGeom = new THREE.BoxGeometry(0.08, 0.09, 0.025);
    palmGeom.translate(0, 0.045, 0);
    const palmSolidMat = new THREE.MeshPhongMaterial({
        color: COL_PALM, transparent: true, opacity: 0.35, shininess: 40
    });
    const palmWireMat = new THREE.MeshBasicMaterial({
        color: COL_PALM, wireframe: true, transparent: true, opacity: 0.45
    });
    wristGroup.add(new THREE.Mesh(palmGeom, palmSolidMat));
    wristGroup.add(new THREE.Mesh(palmGeom, palmWireMat));

    // Wrist joint sphere
    const wristSphere = new THREE.SphereGeometry(0.02, 8, 6);
    const wristMat = new THREE.MeshPhongMaterial({
        color: COL_JOINT, emissive: COL_JOINT, emissiveIntensity: 0.3,
        transparent: true, opacity: 0.5
    });
    wristGroup.add(new THREE.Mesh(wristSphere, wristMat));

    // Palm top anchor (finger roots)
    const palmTop = new THREE.Group();
    palmTop.position.set(0, 0.09, 0);
    wristGroup.add(palmTop);

    // ── Fingers ── (positioned at palm top, spread across X)
    // Finger configs: name, xOffset, segLengths, radius, color
    const fingerConfigs = [
        { name: 'thumb', xOff: -0.042, zOff: 0.012, segs: [0.028, 0.025], r: 0.007, col: COL_FINGER_FREE, rotZ: 0.6 },
        { name: 'index', xOff: -0.022, zOff: 0, segs: [0.03, 0.022, 0.018], r: 0.006, col: COL_FINGER_HOLD },
        { name: 'middle', xOff: -0.002, zOff: 0, segs: [0.033, 0.025, 0.020], r: 0.006, col: COL_FINGER_HOLD },
        { name: 'ring', xOff: 0.018, zOff: 0, segs: [0.030, 0.022, 0.018], r: 0.0055, col: COL_FINGER_FREE },
        { name: 'pinky', xOff: 0.035, zOff: 0, segs: [0.025, 0.018, 0.015], r: 0.005, col: COL_FINGER_FREE },
    ];

    const fingers = {};
    for (const fc of fingerConfigs) {
        const finger = buildFinger(fc.segs, fc.r, fc.col, fc.col);
        finger.root.position.set(fc.xOff, 0, fc.zOff);
        if (fc.rotZ) finger.root.rotation.z = fc.rotZ; // thumb angle
        palmTop.add(finger.root);
        fingers[fc.name] = finger;
    }

    // ── Pen (held between index and middle) ──
    const penGroup = new THREE.Group();
    const penLen = 0.12;
    const penGeom = new THREE.CylinderGeometry(0.003, 0.003, penLen, 4);
    penGeom.translate(0, penLen / 2, 0);
    const penMat = new THREE.MeshPhongMaterial({
        color: COL_PEN_IDLE, emissive: COL_PEN_IDLE, emissiveIntensity: 0.4,
        transparent: true, opacity: 0.8
    });
    const penMesh = new THREE.Mesh(penGeom, penMat);
    penGroup.add(penMesh);

    // Pen tip cone
    const tipCone = new THREE.ConeGeometry(0.005, 0.015, 4);
    tipCone.translate(0, penLen + 0.007, 0);
    const tipMat = new THREE.MeshPhongMaterial({
        color: COL_PEN_IDLE, emissive: COL_PEN_IDLE, emissiveIntensity: 0.6,
        transparent: true, opacity: 0.9
    });
    const tipMesh = new THREE.Mesh(tipCone, tipMat);
    penGroup.add(tipMesh);

    // Pen light
    const tipLight = new THREE.PointLight(COL_PEN_IDLE, 0.4, 0.3);
    tipLight.position.set(0, penLen + 0.01, 0);
    penGroup.add(tipLight);

    // Attach pen to middle finger's tip anchor
    penGroup.position.set(0.01, 0, 0.005);
    penGroup.rotation.z = -0.1; // slight tilt
    fingers.middle.tipAnchor.add(penGroup);

    // ── Trail effect ──
    const MAX_TRAIL = 80;
    const trailPositions = [];
    const trailGeometry = new THREE.BufferGeometry();
    const trailMaterial = new THREE.LineBasicMaterial({
        color: 0x38e67a, transparent: true, opacity: 0.6
    });
    const trailLine = new THREE.Line(trailGeometry, trailMaterial);
    scene.add(trailLine);
    let trailActive = false;

    // ── Quaternion state ──
    const quatState = {
        s1: { target: new THREE.Quaternion(), current: new THREE.Quaternion() },
        s2: { target: new THREE.Quaternion(), current: new THREE.Quaternion() },
        s3: { target: new THREE.Quaternion(), current: new THREE.Quaternion() },
    };

    let penDown = false;
    let isDataActive = false;
    let lastDataTime = 0;
    const statusDot = document.getElementById('handWidgetStatus');

    // ── Animate fingers based on S3 quaternion ──
    function animateFingers(s3Quat) {
        // Extract pitch-like rotation from S3 for finger curl
        const euler = new THREE.Euler();
        euler.setFromQuaternion(s3Quat, 'XYZ');

        // Index and middle follow S3 closely (holding pen)
        const curlAmount = Math.max(-1.0, Math.min(1.0, euler.x * 0.5));
        for (const joint of fingers.index.joints) {
            joint.rotation.x = curlAmount * 0.3;
        }
        for (const joint of fingers.middle.joints) {
            joint.rotation.x = curlAmount * 0.25;
        }

        // Ring, pinky curl slightly more (natural rest pose)
        for (const joint of fingers.ring.joints) {
            joint.rotation.x = curlAmount * 0.35 + 0.15;
        }
        for (const joint of fingers.pinky.joints) {
            joint.rotation.x = curlAmount * 0.4 + 0.2;
        }

        // Thumb slight curl
        for (const joint of fingers.thumb.joints) {
            joint.rotation.x = curlAmount * 0.2 + 0.1;
        }
    }

    // ── Update function (called from app.js) ──
    window.handWidgetUpdate = function (data) {
        isDataActive = true;
        lastDataTime = performance.now();

        if (statusDot) {
            statusDot.textContent = '● LIVE';
            statusDot.style.color = '#4ade80';
        }

        penDown = data.pen || false;

        // Pen color
        const penColor = penDown ? COL_PEN_ACTIVE : COL_PEN_IDLE;
        penMat.color.setHex(penColor);
        penMat.emissive.setHex(penColor);
        tipMat.color.setHex(penColor);
        tipMat.emissive.setHex(penColor);
        tipLight.color.setHex(penColor);
        tipLight.intensity = penDown ? 1.0 : 0.4;

        // Update quaternions
        const qMap = { S1q: 's1', S2q: 's2', S3q: 's3' };
        for (const [key, name] of Object.entries(qMap)) {
            const q = data[key];
            if (q && q.length === 4) {
                quatState[name].target.set(q[1], q[2], q[3], q[0]);
            }
        }

        // Trail
        if (penDown) {
            const tipWorldPos = new THREE.Vector3();
            tipMesh.getWorldPosition(tipWorldPos);
            trailPositions.push(tipWorldPos.clone());
            if (trailPositions.length > MAX_TRAIL) trailPositions.shift();
            trailActive = true;
        } else if (trailActive) {
            trailActive = false;
        }

        if (trailPositions.length > 1) {
            const posArray = new Float32Array(trailPositions.length * 3);
            for (let i = 0; i < trailPositions.length; i++) {
                posArray[i * 3] = trailPositions[i].x;
                posArray[i * 3 + 1] = trailPositions[i].y;
                posArray[i * 3 + 2] = trailPositions[i].z;
            }
            trailGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            trailGeometry.computeBoundingSphere();
            trailLine.visible = true;
        }
    };

    // ── Animation loop ──
    function animate() {
        requestAnimationFrame(animate);

        // Slerp quaternions
        for (const name of ['s1', 's2', 's3']) {
            quatState[name].current.slerp(quatState[name].target, LERP_FACTOR);
        }

        // Apply to hierarchy
        forearmBone.quaternion.copy(quatState.s1.current);
        wristGroup.quaternion.copy(quatState.s2.current);

        // Animate finger curl from S3
        animateFingers(quatState.s3.current);

        // Connection timeout
        if (isDataActive && performance.now() - lastDataTime > 3000) {
            isDataActive = false;
            if (statusDot) {
                statusDot.textContent = '● IDLE';
                statusDot.style.color = '#f59e0b';
            }
        }

        // Idle rotation
        if (!isDataActive) {
            armGroup.rotation.y += 0.003;
        } else {
            armGroup.rotation.y *= 0.95;
        }

        // Fade trail
        if (!trailActive && trailPositions.length > 0) {
            trailMaterial.opacity -= 0.008;
            if (trailMaterial.opacity <= 0) {
                trailPositions.length = 0;
                trailLine.visible = false;
                trailMaterial.opacity = 0.6;
            }
        } else {
            trailMaterial.opacity = 0.6;
        }

        renderer.render(scene, cam);
    }
    animate();

    // ── Widget drag ──
    const container = document.getElementById('handWidgetContainer');
    const header = container?.querySelector('.hand-widget-header');
    if (header && container) {
        let dragX = 0, dragY = 0, isDragging = false;
        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragX = e.clientX - container.offsetLeft;
            dragY = e.clientY - container.offsetTop;
            header.style.cursor = 'grabbing';
        });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            container.style.left = (e.clientX - dragX) + 'px';
            container.style.top = (e.clientY - dragY) + 'px';
            container.style.right = 'auto';
            container.style.bottom = 'auto';
        });
        document.addEventListener('mouseup', () => {
            isDragging = false;
            if (header) header.style.cursor = 'grab';
        });
    }

    // ── Toggle minimize ──
    const toggleBtn = document.getElementById('handWidgetToggle');
    let isMinimized = false;
    if (toggleBtn && container) {
        toggleBtn.addEventListener('click', () => {
            isMinimized = !isMinimized;
            canvas.style.display = isMinimized ? 'none' : 'block';
            toggleBtn.textContent = isMinimized ? '▲' : '▼';
            container.style.height = isMinimized ? '36px' : (WIDGET_HEIGHT + 36) + 'px';
        });
    }

})();
