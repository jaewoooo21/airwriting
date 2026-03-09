// ══════════════════════════════════════════════════════════════
// 3D Hand Mesh Widget — Organic Human Hand with IMU-driven Index
// Uses smooth LatheGeometry profiles for finger phalanges,
// organic palm shape, holographic tech aesthetic.
// Driven by IMU quaternion data: S1(wrist), S2(palm), S3(index finger).
// ══════════════════════════════════════════════════════════════
(function () {
    'use strict';

    const WIDGET_WIDTH = 280;
    const WIDGET_HEIGHT = 210;
    const LERP_FACTOR = 0.25;

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
    const cam = new THREE.PerspectiveCamera(40, WIDGET_WIDTH / WIDGET_HEIGHT, 0.01, 10);
    cam.position.set(0, 0.12, 0.30);
    cam.lookAt(0, 0.06, -0.02);

    // ── Lights ──
    scene.add(new THREE.AmbientLight(0x8899bb, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0.5, 1.0, 0.8);
    scene.add(dirLight);
    const rimLight = new THREE.DirectionalLight(0x4ade80, 0.25);
    rimLight.position.set(-0.5, 0.2, -0.5);
    scene.add(rimLight);

    // ── Subtle grid floor ──
    const gridHelper = new THREE.GridHelper(0.5, 12, 0x1a1a2e, 0x1a1a2e);
    gridHelper.position.y = -0.02;
    gridHelper.material.opacity = 0.25;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);

    // ── Minimal XYZ Axes ──
    const axesHelper = new THREE.AxesHelper(0.15);
    axesHelper.position.set(-0.1, -0.01, -0.1);
    // Darken standard RGB to match tech aesthetic
    axesHelper.material.opacity = 0.6;
    axesHelper.material.transparent = true;
    scene.add(axesHelper);

    // ── Semi-transparent Writing Board ──
    const boardGeom = new THREE.PlaneGeometry(0.6, 0.4);
    const boardMat = new THREE.MeshBasicMaterial({
        color: 0x0ea5e9,
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const writingBoard = new THREE.Mesh(boardGeom, boardMat);
    // Position the board slightly in front of the camera, behind the hand
    writingBoard.position.set(0, 0.1, -0.15);
    
    // Add a glowing border to the board
    const boardEdges = new THREE.EdgesGeometry(boardGeom);
    const borderMat = new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.3 });
    const boardBorder = new THREE.LineSegments(boardEdges, borderMat);
    writingBoard.add(boardBorder);
    scene.add(writingBoard);

    // ── Color palette (holographic tech aesthetic) ──
    const COL_BONE = 0x88aacc;
    const COL_JOINT = 0x5588aa;
    const COL_PALM = 0x7799bb;
    const COL_INDEX = 0x4ade80;    // Index finger highlight (IMU-driven)
    const COL_HOLD = 0x66bbdd;     // Fingers holding pen
    const COL_FREE = 0x7799bb;     // Passive fingers
    const COL_PEN_IDLE = 0x38bdf8;
    const COL_PEN_ACTIVE = 0xff1a66;

    // ── Organic finger phalanx using LatheGeometry (smooth rounded profile) ──
    function createOrganicPhalanx(length, radius, color, isLast) {
        const group = new THREE.Group();
        const steps = 16;

        // Build a smooth profile curve for the phalanx cross-section
        const points = [];
        const halfLen = length;
        // Bottom rounded cap
        for (let i = 0; i <= 4; i++) {
            const a = (Math.PI / 2) * (i / 4);
            const r = radius * Math.cos(a) * 0.9;
            const y = -radius * 0.3 * Math.sin(a);
            points.push(new THREE.Vector2(Math.max(r, 0.0001), y));
        }
        // Main shaft (tapered slightly)
        const tapNum = 8;
        for (let i = 0; i <= tapNum; i++) {
            const t = i / tapNum;
            const r = radius * (1.0 - t * 0.15); // subtle taper
            const y = t * halfLen;
            points.push(new THREE.Vector2(r, y));
        }
        // Top rounded cap (fingertip if last segment)
        if (isLast) {
            for (let i = 1; i <= 6; i++) {
                const a = (Math.PI / 2) * (i / 6);
                const tipR = radius * 0.85;
                const r = tipR * Math.cos(a);
                const y = halfLen + tipR * 0.5 * Math.sin(a);
                points.push(new THREE.Vector2(Math.max(r, 0.0001), y));
            }
        }

        const latheGeom = new THREE.LatheGeometry(points, steps);
        const solidMat = new THREE.MeshPhongMaterial({
            color: color, transparent: true, opacity: 0.45, shininess: 60,
            side: THREE.DoubleSide
        });
        const mesh = new THREE.Mesh(latheGeom, solidMat);
        mesh.castShadow = true;
        group.add(mesh);

        // Subtle wireframe overlay
        const wireMat = new THREE.MeshBasicMaterial({
            color: color, wireframe: true, transparent: true, opacity: 0.25
        });
        group.add(new THREE.Mesh(latheGeom, wireMat));

        // Knuckle joint sphere at base
        const knuckleR = radius * 1.2;
        const sGeom = new THREE.SphereGeometry(knuckleR, 16, 12);
        const sMat = new THREE.MeshPhongMaterial({
            color: COL_JOINT, emissive: COL_JOINT, emissiveIntensity: 0.2,
            transparent: true, opacity: 0.5
        });
        group.add(new THREE.Mesh(sGeom, sMat));

        return group;
    }

    // ── Build a finger chain ──
    function buildFinger(segConfigs, color) {
        const root = new THREE.Group();
        let parent = root;
        const joints = [];

        for (let i = 0; i < segConfigs.length; i++) {
            const cfg = segConfigs[i];
            const isLast = (i === segConfigs.length - 1);
            const bone = createOrganicPhalanx(cfg.len, cfg.r, color, isLast);
            parent.add(bone);
            joints.push(bone);

            const anchor = new THREE.Group();
            anchor.position.set(0, cfg.len, 0);
            bone.add(anchor);
            parent = anchor;
        }

        return { root, joints, tipAnchor: parent };
    }

    // ── Organic palm using custom BufferGeometry ──
    function createOrganicPalm() {
        const group = new THREE.Group();

        // Main palm body — ellipsoid-like shape via a scaled sphere
        const palmGeom = new THREE.SphereGeometry(0.045, 24, 16);
        // Scale to flat oval palm shape
        palmGeom.scale(0.95, 1.1, 0.30);
        palmGeom.translate(0, 0.048, 0);
        const palmMat = new THREE.MeshPhongMaterial({
            color: COL_PALM, transparent: true, opacity: 0.4, shininess: 50,
            side: THREE.DoubleSide
        });
        const palmMesh = new THREE.Mesh(palmGeom, palmMat);
        group.add(palmMesh);

        // Palm wireframe
        const palmWire = new THREE.MeshBasicMaterial({
            color: COL_PALM, wireframe: true, transparent: true, opacity: 0.2
        });
        group.add(new THREE.Mesh(palmGeom, palmWire));

        // Thenar eminence (thumb-side muscle pad) — RIGHT hand: positive X
        const thenarGeom = new THREE.SphereGeometry(0.018, 14, 10);
        thenarGeom.scale(1.1, 1.3, 0.7);
        thenarGeom.translate(0.030, 0.035, 0.007);
        const thenarMat = new THREE.MeshPhongMaterial({
            color: COL_PALM, transparent: true, opacity: 0.35, shininess: 40
        });
        group.add(new THREE.Mesh(thenarGeom, thenarMat));

        // Hypothenar eminence (pinky-side muscle pad) — RIGHT hand: negative X
        const hypoGeom = new THREE.SphereGeometry(0.015, 14, 10);
        hypoGeom.scale(0.9, 1.2, 0.6);
        hypoGeom.translate(-0.026, 0.035, 0.007);
        group.add(new THREE.Mesh(hypoGeom, thenarMat));

        // Knuckle ridge across the top of the palm
        for (let i = 0; i < 4; i++) {
            const knGeom = new THREE.SphereGeometry(0.006, 12, 8);
            knGeom.translate(0.022 - i * 0.015, 0.092, 0.003);
            const knMat = new THREE.MeshPhongMaterial({
                color: COL_JOINT, emissive: COL_JOINT, emissiveIntensity: 0.15,
                transparent: true, opacity: 0.45
            });
            group.add(new THREE.Mesh(knGeom, knMat));
        }

        return group;
    }

    // ══════════════════════════════════════════
    // Build the hand hierarchy
    // ══════════════════════════════════════════
    const armGroup = new THREE.Group();
    armGroup.position.set(0, -0.04, 0);
    scene.add(armGroup);

    // S1: Forearm pivot (attached at wrist backward)
    const forearmBone = new THREE.Group();
    armGroup.add(forearmBone);
    
    // Add visual forearm extending backward from wrist
    // The forearm is about 15-20cm, we use 0.18m in this scale
    const armLen = 0.18;
    const armGeom = new THREE.CylinderGeometry(0.018, 0.015, armLen, 16);
    // Cylinder default is along Y-axis, centered. We want the wrist (top) to be at origin.
    // So we translate the cylinder down by armLen / 2.
    armGeom.translate(0, -armLen / 2, 0); 
    const armMat = new THREE.MeshPhongMaterial({
        color: COL_BONE, transparent: true, opacity: 0.35, shininess: 30
    });
    const armMesh = new THREE.Mesh(armGeom, armMat);
    forearmBone.add(armMesh);
    
    // Add elbow joint visualization at the bottom of the forearm
    const elbowGeom = new THREE.SphereGeometry(0.018, 16, 12);
    elbowGeom.translate(0, -armLen, 0);
    const elbowMat = new THREE.MeshPhongMaterial({
        color: COL_JOINT, emissive: COL_JOINT, emissiveIntensity: 0.25,
        transparent: true, opacity: 0.5
    });
    forearmBone.add(new THREE.Mesh(elbowGeom, elbowMat));

    const forearmEnd = new THREE.Group();
    forearmEnd.position.set(0, 0, 0);
    forearmBone.add(forearmEnd);

    // S2: Wrist + Palm
    const wristGroup = new THREE.Group();
    forearmEnd.add(wristGroup);

    // Organic palm
    const palmMesh = createOrganicPalm();
    wristGroup.add(palmMesh);

    // Wrist joint
    const wristSGeom = new THREE.SphereGeometry(0.015, 16, 12);
    wristSGeom.scale(1.5, 0.5, 0.8);
    const wristSMat = new THREE.MeshPhongMaterial({
        color: COL_JOINT, emissive: COL_JOINT, emissiveIntensity: 0.25,
        transparent: true, opacity: 0.5
    });
    wristGroup.add(new THREE.Mesh(wristSGeom, wristSMat));

    // Palm top anchor
    const palmTop = new THREE.Group();
    palmTop.position.set(0, 0.095, 0);
    wristGroup.add(palmTop);

    // ── Finger definitions ──
    const fingerDefs = [
        {
            name: 'thumb', xOff: 0.042, zOff: 0.014, rotZ: -0.55, color: COL_FREE,
            segs: [{ len: 0.026, r: 0.0075 }, { len: 0.022, r: 0.007 }]
        },
        {
            name: 'index', xOff: 0.024, zOff: 0, color: COL_INDEX, // highlighted — IMU driven
            segs: [{ len: 0.030, r: 0.0065 }, { len: 0.021, r: 0.006 }, { len: 0.017, r: 0.0055 }]
        },
        {
            name: 'middle', xOff: 0.004, zOff: 0, color: COL_HOLD,
            segs: [{ len: 0.034, r: 0.0068 }, { len: 0.024, r: 0.006 }, { len: 0.019, r: 0.0055 }]
        },
        {
            name: 'ring', xOff: -0.016, zOff: 0, color: COL_FREE,
            segs: [{ len: 0.031, r: 0.006 }, { len: 0.021, r: 0.0055 }, { len: 0.017, r: 0.005 }]
        },
        {
            name: 'pinky', xOff: -0.034, zOff: 0, color: COL_FREE,
            segs: [{ len: 0.025, r: 0.0052 }, { len: 0.017, r: 0.0045 }, { len: 0.014, r: 0.004 }]
        },
    ];

    const fingers = {};
    for (const fd of fingerDefs) {
        const finger = buildFinger(fd.segs, fd.color);
        finger.root.position.set(fd.xOff, 0, fd.zOff);
        if (fd.rotZ) finger.root.rotation.z = fd.rotZ;
        palmTop.add(finger.root);
        fingers[fd.name] = finger;
    }

    // ── Pen (sleek stylus held between index and middle) ──
    const penGroup = new THREE.Group();
    const penLen = 0.10;
    const penGeom = new THREE.CylinderGeometry(0.0025, 0.003, penLen, 8);
    penGeom.translate(0, penLen / 2, 0);
    const penMat = new THREE.MeshPhongMaterial({
        color: COL_PEN_IDLE, emissive: COL_PEN_IDLE, emissiveIntensity: 0.4,
        transparent: true, opacity: 0.8
    });
    const penMesh = new THREE.Mesh(penGeom, penMat);
    penGroup.add(penMesh);

    const tipCone = new THREE.ConeGeometry(0.004, 0.012, 8);
    tipCone.translate(0, penLen + 0.006, 0);
    const tipMat = new THREE.MeshPhongMaterial({
        color: COL_PEN_IDLE, emissive: COL_PEN_IDLE, emissiveIntensity: 0.6,
        transparent: true, opacity: 0.9
    });
    const tipMesh = new THREE.Mesh(tipCone, tipMat);
    penGroup.add(tipMesh);

    const tipLight = new THREE.PointLight(COL_PEN_IDLE, 0.4, 0.3);
    tipLight.position.set(0, penLen + 0.01, 0);
    penGroup.add(tipLight);

    penGroup.position.set(0.008, 0, 0.004);
    penGroup.rotation.z = -0.08;
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

    // ── Animate fingers based on quaternions ──
    function animateFingers(s3Quat) {
        // Extract euler angles from S3 (index finger IMU)
        const euler = new THREE.Euler();
        euler.setFromQuaternion(s3Quat, 'XYZ');

        // ── Index finger: directly driven by S3 IMU ──
        // S3 pitch (euler.x) → proximal joint (main curl)
        // S3 yaw (euler.z) → lateral splay
        const indexCurl = euler.x; // Direct mapping from IMU
        const indexSplay = euler.z * 0.3;

        if (fingers.index.joints.length >= 3) {
            // Proximal phalanx — main bend from IMU pitch
            fingers.index.joints[0].rotation.x = indexCurl * 0.6;
            fingers.index.joints[0].rotation.z = indexSplay;
            // Intermediate phalanx — follow-through
            fingers.index.joints[1].rotation.x = indexCurl * 0.45;
            // Distal phalanx — fine tip curl
            fingers.index.joints[2].rotation.x = indexCurl * 0.3;
        }

        // ── Other fingers: subtle response based on S3 with natural offsets ──
        const curlBase = Math.max(-1.0, Math.min(1.0, euler.x * 0.3));

        // Middle (holding pen, less curl)
        for (let i = 0; i < fingers.middle.joints.length; i++) {
            fingers.middle.joints[i].rotation.x = curlBase * (0.2 + i * 0.05);
        }

        // Ring (slightly more curled naturally)
        for (let i = 0; i < fingers.ring.joints.length; i++) {
            fingers.ring.joints[i].rotation.x = curlBase * 0.3 + 0.15 + i * 0.05;
        }

        // Pinky (most curled)
        for (let i = 0; i < fingers.pinky.joints.length; i++) {
            fingers.pinky.joints[i].rotation.x = curlBase * 0.35 + 0.2 + i * 0.06;
        }

        // Thumb (opposing grip)
        for (let i = 0; i < fingers.thumb.joints.length; i++) {
            fingers.thumb.joints[i].rotation.x = curlBase * 0.2 + 0.1;
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

        // Animate fingers — index is IMU-driven via S3
        animateFingers(quatState.s3.current);

        // Connection timeout
        if (isDataActive && performance.now() - lastDataTime > 3000) {
            isDataActive = false;
            if (statusDot) {
                statusDot.textContent = '● IDLE';
                statusDot.style.color = '#f59e0b';
            }
        }

        // Idle / Active rotation
        if (!isDataActive) {
            armGroup.rotation.y += 0.004;
            armGroup.rotation.x += (0 - armGroup.rotation.x) * 0.05;
        } else {
            armGroup.rotation.x += (-1.57 - armGroup.rotation.x) * 0.1;
            armGroup.rotation.y += (0 - armGroup.rotation.y) * 0.1;
            armGroup.rotation.z += (0 - armGroup.rotation.z) * 0.1;
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
