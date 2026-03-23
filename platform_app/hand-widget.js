(function () {
    'use strict';

    if (typeof THREE === 'undefined') return;

    const mainCanvas = document.getElementById('mainHandCanvas');
    const widgetCanvas = document.getElementById('handWidgetCanvas');
    if (!mainCanvas && !widgetCanvas) return;

    const statusDot = document.getElementById('handWidgetStatus');
    const trailLength = 24;
    const upAxis = new THREE.Vector3(0, 1, 0);

    const sharedState = {
        active: false,
        penDown: false,
        lastDataTime: 0,
        target: {
            s1: new THREE.Quaternion(),
            s2: new THREE.Quaternion(),
            s3: new THREE.Quaternion(),
            pos: new THREE.Vector3(),
            vel: new THREE.Vector3(),
            direction: new THREE.Vector3(0.0, 1.0, 0.15).normalize(),
        },
        current: {
            s1: new THREE.Quaternion(),
            s2: new THREE.Quaternion(),
            s3: new THREE.Quaternion(),
            pos: new THREE.Vector3(),
            vel: new THREE.Vector3(),
            direction: new THREE.Vector3(0.0, 1.0, 0.15).normalize(),
        },
    };

    function createFinger(material, segmentLengths, radius) {
        const root = new THREE.Group();
        const joints = [];
        let parent = root;

        segmentLengths.forEach((length, idx) => {
            const joint = new THREE.Group();
            parent.add(joint);

            const geom = new THREE.CylinderGeometry(radius * 0.9, radius, length, 12);
            geom.translate(0, length / 2, 0);
            const mesh = new THREE.Mesh(geom, material.clone());
            mesh.castShadow = true;
            joint.add(mesh);

            const knuckle = new THREE.Mesh(
                new THREE.SphereGeometry(radius * 1.05, 12, 10),
                material.clone()
            );
            joint.add(knuckle);

            joints.push(joint);

            const anchor = new THREE.Group();
            anchor.position.y = length;
            joint.add(anchor);
            parent = anchor;

            if (idx === segmentLengths.length - 1) {
                const tip = new THREE.Mesh(
                    new THREE.SphereGeometry(radius * 0.85, 10, 8),
                    material.clone()
                );
                tip.position.y = 0.01;
                parent.add(tip);
            }
        });

        return { root, joints, tipAnchor: parent };
    }

    function createWidgetRig(scene) {
        const rigRoot = new THREE.Group();
        scene.add(rigRoot);

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));

        const key = new THREE.DirectionalLight(0xffffff, 0.9);
        key.position.set(1.2, 1.6, 1.0);
        scene.add(key);

        const rim = new THREE.DirectionalLight(0x38bdf8, 0.35);
        rim.position.set(-1.0, 0.6, -1.0);
        scene.add(rim);

        const armMaterial = new THREE.MeshStandardMaterial({
            color: 0x79a8c9,
            roughness: 0.35,
            metalness: 0.15,
            transparent: true,
            opacity: 0.8,
        });

        const jointMaterial = new THREE.MeshStandardMaterial({
            color: 0xcfe8ff,
            roughness: 0.2,
            metalness: 0.25,
            transparent: true,
            opacity: 0.85,
            emissive: new THREE.Color(0x0ea5e9),
            emissiveIntensity: 0.12,
        });

        const fingerMaterial = new THREE.MeshStandardMaterial({
            color: 0x8dd3b5,
            roughness: 0.25,
            metalness: 0.15,
            transparent: true,
            opacity: 0.85,
        });

        const forearm = new THREE.Group();
        rigRoot.add(forearm);

        const forearmMesh = new THREE.Mesh(
            new THREE.CylinderGeometry(0.04, 0.05, 0.5, 18),
            armMaterial
        );
        forearmMesh.rotation.z = 0.05;
        forearmMesh.position.y = -0.25;
        forearm.add(forearmMesh);

        const elbow = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 14), jointMaterial);
        elbow.position.y = -0.5;
        forearm.add(elbow);

        const wrist = new THREE.Group();
        forearm.add(wrist);

        const palm = new THREE.Mesh(
            new THREE.BoxGeometry(0.24, 0.18, 0.09),
            armMaterial.clone()
        );
        palm.position.set(0, 0.1, 0);
        palm.scale.set(1.0, 1.0, 0.7);
        wrist.add(palm);

        const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 14), jointMaterial);
        wrist.add(wristJoint);

        const palmTop = new THREE.Group();
        palmTop.position.set(0, 0.18, 0);
        wrist.add(palmTop);

        const fingerDefs = [
            { key: 'thumb', offset: [0.11, -0.03, 0.03], rotZ: -0.75, lens: [0.08, 0.06], r: 0.02 },
            { key: 'index', offset: [0.07, 0.0, 0.01], rotZ: -0.08, lens: [0.1, 0.08, 0.06], r: 0.017 },
            { key: 'middle', offset: [0.025, 0.0, 0.0], rotZ: -0.02, lens: [0.11, 0.09, 0.065], r: 0.018 },
            { key: 'ring', offset: [-0.02, 0.0, -0.005], rotZ: 0.04, lens: [0.1, 0.08, 0.06], r: 0.016 },
            { key: 'pinky', offset: [-0.07, 0.0, -0.01], rotZ: 0.12, lens: [0.08, 0.06, 0.05], r: 0.014 },
        ];

        const fingers = {};
        fingerDefs.forEach(def => {
            const finger = createFinger(fingerMaterial, def.lens, def.r);
            finger.root.position.set(def.offset[0], def.offset[1], def.offset[2]);
            finger.root.rotation.z = def.rotZ;
            palmTop.add(finger.root);
            fingers[def.key] = finger;
        });

        const penGroup = new THREE.Group();
        const penBody = new THREE.Mesh(
            new THREE.CylinderGeometry(0.008, 0.01, 0.32, 10),
            new THREE.MeshStandardMaterial({
                color: 0x38bdf8,
                emissive: new THREE.Color(0x0ea5e9),
                emissiveIntensity: 0.6,
                roughness: 0.15,
                metalness: 0.4,
            })
        );
        penBody.position.y = 0.16;
        penGroup.add(penBody);

        const penTip = new THREE.Mesh(
            new THREE.ConeGeometry(0.014, 0.05, 10),
            new THREE.MeshStandardMaterial({
                color: 0xf8fafc,
                emissive: new THREE.Color(0xffffff),
                emissiveIntensity: 0.2,
            })
        );
        penTip.position.y = 0.34;
        penGroup.add(penTip);
        penGroup.rotation.z = -0.18;
        penGroup.position.set(0.02, 0.01, 0.015);
        fingers.middle.tipAnchor.add(penGroup);

        rigRoot.position.set(0, -0.02, 0);
        rigRoot.scale.setScalar(0.95);

        return { rigRoot, forearm, wrist, fingers, penBody };
    }

    function createWidgetView(canvas) {
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 20);
        const rig = createWidgetRig(scene);

        function resize() {
            const width = canvas.clientWidth || 280;
            const height = canvas.clientHeight || 210;
            renderer.setSize(width, height, false);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            camera.aspect = width / Math.max(height, 1);
            camera.updateProjectionMatrix();
            camera.position.set(0.0, 0.14, 0.62);
            camera.lookAt(0.0, 0.1, 0.0);
        }

        resize();
        return { type: 'widget', renderer, scene, camera, rig, resize };
    }

    function createMainView(canvas) {
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setClearColor(0x000000, 0);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 20);

        scene.add(new THREE.AmbientLight(0xf8fbff, 1.1));

        const key = new THREE.DirectionalLight(0xffffff, 1.2);
        key.position.set(1.6, 1.8, 2.2);
        scene.add(key);

        const fill = new THREE.DirectionalLight(0x7dd3fc, 0.55);
        fill.position.set(-1.8, 0.2, 1.2);
        scene.add(fill);

        const stage = new THREE.Group();
        stage.position.set(0, 0.03, 0);
        scene.add(stage);

        const penGroup = new THREE.Group();
        stage.add(penGroup);

        const penShaft = new THREE.Mesh(
            new THREE.CylinderGeometry(0.012, 0.015, 0.36, 12),
            new THREE.MeshStandardMaterial({
                color: 0x38bdf8,
                emissive: new THREE.Color(0x0ea5e9),
                emissiveIntensity: 0.35,
                roughness: 0.18,
                metalness: 0.42,
            })
        );
        penShaft.position.y = 0.17;
        penGroup.add(penShaft);

        const penTip = new THREE.Mesh(
            new THREE.ConeGeometry(0.022, 0.08, 12),
            new THREE.MeshStandardMaterial({
                color: 0xf8fafc,
                emissive: new THREE.Color(0xffffff),
                emissiveIntensity: 0.18,
                roughness: 0.12,
                metalness: 0.08,
            })
        );
        penTip.position.y = 0.38;
        penGroup.add(penTip);

        const penHalo = new THREE.Mesh(
            new THREE.SphereGeometry(0.032, 18, 14),
            new THREE.MeshBasicMaterial({
                color: 0x67e8f9,
                transparent: true,
                opacity: 0.22,
                depthWrite: false,
            })
        );
        penHalo.position.y = 0.39;
        penGroup.add(penHalo);

        const trailPoints = Array.from({ length: trailLength }, () => new THREE.Vector3());
        const trailGeometry = new THREE.BufferGeometry().setFromPoints(trailPoints);
        const trailLine = new THREE.Line(
            trailGeometry,
            new THREE.LineBasicMaterial({
                color: 0x38bdf8,
                transparent: true,
                opacity: 0.5,
            })
        );
        stage.add(trailLine);

        function resize() {
            const width = canvas.clientWidth || 1280;
            const height = canvas.clientHeight || 720;
            renderer.setSize(width, height, false);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            camera.aspect = width / Math.max(height, 1);
            camera.updateProjectionMatrix();
            camera.position.set(0, 0.1, 2.45);
            camera.lookAt(0, 0.05, 0);
        }

        resize();
        return {
            type: 'main',
            renderer,
            scene,
            camera,
            stage,
            penGroup,
            penShaft,
            penHalo,
            trailPoints,
            trailGeometry,
            resize,
        };
    }

    function updateStatus() {
        if (!statusDot) return;
        if (sharedState.active) {
            statusDot.textContent = 'LIVE';
            statusDot.style.color = '#4ade80';
        } else {
            statusDot.textContent = 'IDLE';
            statusDot.style.color = '#f59e0b';
        }
    }

    function mapStagePosition(pos) {
        return new THREE.Vector3(
            THREE.MathUtils.clamp(pos.x * 6.5, -0.6, 0.6),
            THREE.MathUtils.clamp(pos.y * 5.2, -0.38, 0.38),
            THREE.MathUtils.clamp(0.18 - pos.z * 4.0, 0.06, 0.42)
        );
    }

    function updateFallbackDirection() {
        const velocity = sharedState.target.vel;
        if (velocity.lengthSq() > 1e-6) {
            sharedState.target.direction.copy(velocity).normalize();
            return;
        }

        const dir = new THREE.Vector3(0, 1, 0.22).applyQuaternion(sharedState.target.s3);
        if (dir.lengthSq() > 1e-6) {
            sharedState.target.direction.copy(dir.normalize());
        }
    }

    function applyWidgetPose(rig) {
        rig.forearm.quaternion.copy(sharedState.current.s1);
        rig.wrist.quaternion.copy(sharedState.current.s2);

        const s3Euler = new THREE.Euler().setFromQuaternion(sharedState.current.s3, 'XYZ');
        const curl = THREE.MathUtils.clamp(s3Euler.x, -1.2, 1.2);
        const yaw = THREE.MathUtils.clamp(s3Euler.z, -0.7, 0.7);

        const index = rig.fingers.index.joints;
        if (index[0]) index[0].rotation.set(curl * 0.75, 0, yaw * 0.25);
        if (index[1]) index[1].rotation.x = curl * 0.55;
        if (index[2]) index[2].rotation.x = curl * 0.35;

        rig.fingers.middle.joints.forEach((joint, idx) => {
            joint.rotation.x = curl * (0.35 - idx * 0.06) + 0.08;
        });
        rig.fingers.ring.joints.forEach((joint, idx) => {
            joint.rotation.x = curl * (0.28 - idx * 0.05) + 0.16;
        });
        rig.fingers.pinky.joints.forEach((joint, idx) => {
            joint.rotation.x = curl * (0.22 - idx * 0.04) + 0.22;
        });
        rig.fingers.thumb.joints.forEach((joint, idx) => {
            joint.rotation.x = 0.1 + curl * (0.15 - idx * 0.03);
        });

        const penColor = sharedState.penDown ? 0xff5a7a : 0x38bdf8;
        rig.penBody.material.color.setHex(penColor);
        rig.penBody.material.emissive.setHex(sharedState.penDown ? 0xff355d : 0x0ea5e9);
    }

    function applyMainPose(view, now) {
        const targetPos = mapStagePosition(sharedState.current.pos);
        view.penGroup.position.copy(targetPos);

        const direction = sharedState.current.direction.clone();
        if (direction.lengthSq() < 1e-6) {
            direction.set(0, 1, 0.15);
        }
        direction.normalize();
        view.penGroup.quaternion.setFromUnitVectors(upAxis, direction);

        const tipColor = sharedState.penDown ? 0xff6b8d : 0x38bdf8;
        const emissive = sharedState.penDown ? 0xff355d : 0x0ea5e9;
        view.penShaft.material.color.setHex(tipColor);
        view.penShaft.material.emissive.setHex(emissive);
        view.penHalo.material.color.setHex(sharedState.penDown ? 0xff8aa5 : 0x67e8f9);
        view.penHalo.material.opacity = sharedState.penDown ? 0.34 : 0.18;

        view.trailPoints.shift();
        view.trailPoints.push(targetPos.clone());
        view.trailGeometry.setFromPoints(view.trailPoints);

        if (!sharedState.active) {
            const t = now * 0.00045;
            view.penGroup.position.set(Math.sin(t) * 0.18, Math.cos(t * 0.7) * 0.12, 0.22);
            view.penGroup.quaternion.setFromUnitVectors(
                upAxis,
                new THREE.Vector3(Math.sin(t) * 0.2, 1, 0.15).normalize()
            );
            view.stage.rotation.y = Math.sin(t * 0.5) * 0.02;
        } else {
            view.stage.rotation.y += (0 - view.stage.rotation.y) * 0.08;
        }
    }

    const views = [];
    if (widgetCanvas) views.push(createWidgetView(widgetCanvas));

    window.handWidgetUpdate = function (data) {
        sharedState.active = true;
        sharedState.lastDataTime = performance.now();
        sharedState.penDown = !!data.pen;

        const qMap = { S1q: 's1', S2q: 's2', S3q: 's3' };
        for (const [key, name] of Object.entries(qMap)) {
            const q = data[key];
            if (q && q.length === 4) {
                sharedState.target[name].set(q[1], q[2], q[3], q[0]);
            }
        }

        if (data.pos) {
            sharedState.target.pos.set(
                Number(data.pos.x) || 0,
                Number(data.pos.y) || 0,
                Number(data.pos.z) || 0
            );
        }

        if (data.vel) {
            sharedState.target.vel.set(
                Number(data.vel.x) || 0,
                Number(data.vel.y) || 0,
                Number(data.vel.z) || 0
            );
        }

        updateFallbackDirection();
        updateStatus();
    };

    function animate() {
        requestAnimationFrame(animate);

        const now = performance.now();
        if (sharedState.active && now - sharedState.lastDataTime > 3000) {
            sharedState.active = false;
            updateStatus();
        }

        ['s1', 's2', 's3'].forEach(key => {
            sharedState.current[key].slerp(sharedState.target[key], 0.18);
        });
        sharedState.current.pos.lerp(sharedState.target.pos, 0.22);
        sharedState.current.vel.lerp(sharedState.target.vel, 0.18);
        sharedState.current.direction.lerp(sharedState.target.direction, 0.2).normalize();

        views.forEach(view => {
            if (view.type === 'main') {
                applyMainPose(view, now);
            } else {
                applyWidgetPose(view.rig);
                if (!sharedState.active) {
                    view.rig.rigRoot.rotation.y += 0.004;
                } else {
                    view.rig.rigRoot.rotation.y += (0 - view.rig.rigRoot.rotation.y) * 0.08;
                    view.rig.rigRoot.rotation.x += (0 - view.rig.rigRoot.rotation.x) * 0.08;
                }
            }
            view.renderer.render(view.scene, view.camera);
        });
    }

    animate();

    window.addEventListener('resize', () => {
        views.forEach(view => view.resize());
    });

    const container = document.getElementById('handWidgetContainer');
    const header = container?.querySelector('.hand-widget-header');
    if (header && container) {
        let dragX = 0;
        let dragY = 0;
        let isDragging = false;

        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragX = e.clientX - container.offsetLeft;
            dragY = e.clientY - container.offsetTop;
            header.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            container.style.left = `${e.clientX - dragX}px`;
            container.style.top = `${e.clientY - dragY}px`;
            container.style.right = 'auto';
            container.style.bottom = 'auto';
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            header.style.cursor = 'grab';
        });
    }

    const toggleBtn = document.getElementById('handWidgetToggle');
    if (toggleBtn && container && widgetCanvas) {
        let isMinimized = false;
        toggleBtn.addEventListener('click', () => {
            isMinimized = !isMinimized;
            widgetCanvas.style.display = isMinimized ? 'none' : 'block';
            toggleBtn.textContent = isMinimized ? '+' : '-';
            container.style.height = isMinimized ? '40px' : '';
        });
    }
})();
