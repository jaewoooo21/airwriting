
const translations = {
    ko: {
        introKicker: 'AIRWRITING PLATFORM',
        introTitle: '공중의 움직임을 읽히는 문자 경험으로 바꿉니다',
        introSubtitle: '3개의 IMU 센서, 실시간 추정, 브라우저 시각화를 한 화면에 정리한 AirWriting 공개 데모입니다.',
        introSkip: '바로 들어가기',
        brandTitle: '배포형 AirWriting 플랫폼',
        navHome: '홈',
        navStudio: '스튜디오',
        navAndroid: '안드로이드 연결',
        navTechnology: '기술 개요',
        navTeam: '팀',
        navContact: '문의',
        heroEyebrow: 'Public Demo / Product Surface',
        heroTitle: '손짓을 데이터로 끝내지 않고, 실제로 읽히는 글씨 경험까지 연결합니다.',
        heroBody: 'AirWriting은 IMU 센서 기반 공중 필기를 안정화하고, 브라우저에서 데모·라이브·설치 가이드를 함께 제공하는 공개용 플랫폼입니다. 배포 사이트에서도 바로 이해되고, 로컬 환경에서는 실제 기기와 연결할 수 있게 설계했습니다.',
        heroCtaDemo: '데모 스튜디오 열기',
        heroCtaLive: '라이브 스튜디오 열기',
        heroCtaAndroid: '안드로이드 설치/연결',
        heroDemoLabel: 'DEPLOY MODE',
        heroDemoTitle: '하드웨어 없이 바로 보는 데모',
        heroDemoBody: '배포 환경에서는 기본적으로 시연용 획 데이터를 재생해 실제 글자처럼 읽히는 흐름을 보여줍니다.',
        heroLiveLabel: 'LOCAL MODE',
        heroLiveTitle: '센서 연결 시 라이브 전환',
        heroLiveBody: '로컬 PC와 Android 앱을 같은 네트워크에 두면 WebSocket 기반 실시간 연결을 시도할 수 있습니다.',
        heroSurfaceLabel: 'Readable Demo Strokes',
        heroSurfaceBody: '획이 흩어지지 않도록 평면 보드 위에 선명한 문자 경로로 투영합니다.',
        metricAccuracy: '최고 인식 정확도',
        metricRealtime: '실시간 샘플 처리율',
        metricSensors: '사용 센서 수',
        proofEyebrow: 'Why This Feels Like A Product',
        proofTitle: '공개 배포에서도 가치가 바로 보이도록 구성했습니다',
        proof1Label: '01',
        proof1Title: '읽히는 데모 필기',
        proof1Body: '아무렇게나 흔들리는 궤적이 아니라 문자 형태가 보이는 시연 경로를 기본 제공해 첫인상을 개선했습니다.',
        proof2Label: '02',
        proof2Title: '라이브 연결 분리',
        proof2Body: '공개 배포와 로컬 실험 환경을 구분해, 연결이 없어도 화면이 비어 보이지 않도록 구성했습니다.',
        proof3Label: '03',
        proof3Title: '설치부터 연결까지',
        proof3Body: 'Android 앱 설치, 개발자 옵션, QR/IP pairing, 문제 해결 흐름을 한 페이지에서 안내합니다.',
        flowEyebrow: 'How It Works',
        flowTitle: '사용자는 3단계만 이해하면 됩니다',
        flow1Title: '센서와 필기 궤적 수집',
        flow1Body: '3개의 IMU 센서가 팔과 손의 움직임을 샘플링하고, 필기 구간을 분리합니다.',
        flow2Title: '보정 및 자세 추정',
        flow2Body: '드리프트 억제와 자세 추정으로 공중 움직임을 평면 보드 위 문자 궤적으로 재정렬합니다.',
        flow3Title: '시각화와 액션 연결',
        flow3Body: '브라우저는 인식 결과와 파형을 보여주고, Android 앱은 액션 실행 엔드포인트가 됩니다.',
        pathsEyebrow: 'Choose Your Path',
        pathsTitle: '목적에 따라 바로 들어갈 수 있도록 진입점을 분리했습니다',
        pathsDemoTitle: '배포 데모',
        pathsDemoBody: '하드웨어가 없어도 작동하는 문자 데모와 UI 흐름을 확인합니다.',
        pathsDemoButton: '데모 보기',
        pathsLiveTitle: '로컬 라이브',
        pathsLiveBody: 'PC에서 relay/action dispatcher를 실행한 뒤 실제 센서 스트림을 연결합니다.',
        pathsLiveButton: '라이브 모드 시도',
        studioEyebrow: 'Studio Surface',
        studioTitle: '실시간/데모 공중 필기 스튜디오',
        modeDemo: '데모 모드',
        modeLive: '라이브 모드',
        recognitionLabel: 'Recognition',
        sessionLabel: 'Session Control',
        sessionSafe: '배포 기본값은 데모',
        sessionRunDemo: '데모 다시 재생',
        sessionTryLive: '라이브 연결 시도',
        sessionBackDemo: '데모로 복귀',
        guideLabel: 'Quick Guide',
        guideWhenLive: '라이브 연결 전 체크',
        guideStep1: 'PC와 Android 기기를 같은 Wi-Fi에 연결합니다.',
        guideStep2: 'PC에서 relay/action dispatcher가 실행 중인지 확인합니다.',
        guideStep3: '연결이 실패하면 데모 모드로 돌아가 UI와 인식 흐름을 먼저 확인합니다.',
        boardCaption: '평면 보드에 투영한 읽기 쉬운 데모 스트로크',
        writingState: 'WRITING',
        statusConnection: '연결',
        statusPointer: '포인터',
        statusZupt: '상태',
        statusWord: '현재 단어',
        waveLabel: 'Waveform',
        waveTitle: 'Pitch / Roll / Yaw',
        deployLabel: 'Public Deployment',
        deployTitle: '배포 환경 동작 정책',
        deployBody: 'Render 공개 배포에서는 데모 모드가 기본값입니다. 라이브 모드는 로컬 장비와 relay가 준비된 경우에만 사용하세요.',
        androidEyebrow: 'Android Companion',
        androidTitle: '안드로이드 앱 설치와 Pairing 가이드',
        androidBody: '웹은 시각화와 설정을 담당하고, Android 앱은 실제 액션 실행기 역할을 합니다. 공개 배포에서는 구조를 설명하고, 로컬 환경에서는 실제 IP/QR 연결을 안내합니다.',
        androidCard1Label: 'APP ROLE',
        androidCard1Title: '액션 실행 엔드포인트',
        androidCard1Body: '웹에서 설정한 동작을 Android 인텐트 또는 앱 실행으로 연결하는 companion 앱입니다.',
        androidCard2Label: 'PAIRING',
        androidCard2Title: 'QR 또는 IP 입력 연결',
        androidCard2Body: '같은 Wi-Fi 환경에서 PC 주소를 스캔하거나 직접 입력해 WebSocket으로 연결합니다.',
        androidCard3Label: 'SERVICE',
        androidCard3Title: '서비스형 온보딩',
        androidCard3Body: '설치, 개발자 옵션, 디버깅 허용, 네트워크 체크를 한 페이지에서 정리해 실제 사용자 흐름처럼 구성했습니다.',
        pairLabel: 'Pairing',
        pairUrlLabel: '연결 대상',
        pairStateLabel: '호스트 상태',
        pairRefresh: '주소 새로고침',
        pairOpenStudio: '스튜디오 열기',
        installGuideLabel: 'Install Guide',
        installGuideTitle: '실제 사용자를 위한 설치 순서',
        install1Title: 'Android 앱 열기 또는 설치 준비',
        install1Body: 'Android Studio에서 `android_app` 모듈을 실행하거나, 향후 배포될 APK를 준비합니다.',
        install2Title: '개발자 옵션 켜기',
        install2Body: '휴대폰 설정에서 빌드 번호를 여러 번 눌러 개발자 옵션을 활성화합니다.',
        install3Title: 'USB 디버깅 허용',
        install3Body: 'Android Studio로 직접 설치할 경우 USB 디버깅을 켜고 PC를 신뢰하도록 허용합니다.',
        install4Title: '알 수 없는 앱 설치 허용',
        install4Body: 'APK 파일로 설치할 경우 브라우저 또는 파일 관리자에 대해 설치 권한을 허용합니다.',
        install5Title: '같은 Wi-Fi에 연결',
        install5Body: 'PC와 휴대폰이 동일한 로컬 네트워크에 있어야 `ws://<PC_IP>:18800` 연결이 가능합니다.',
        install6Title: 'QR 스캔 또는 IP 입력',
        install6Body: '앱의 Pairing 화면에서 QR을 스캔하거나 표시된 IP를 직접 입력해 연결합니다.',
        trouble1Title: 'QR은 보이는데 연결이 안 될 때',
        trouble1Body: '휴대폰과 PC가 같은 공유기에 붙어 있는지, PC 방화벽이 18800 포트를 막고 있지 않은지 먼저 확인하세요.',
        trouble2Title: '공개 배포 주소와 PC 주소는 다릅니다',
        trouble2Body: 'Render 주소는 클라우드 서버입니다. 실제 폰은 로컬에서 실행 중인 PC relay 주소에 연결되어야 합니다.',
        trouble3Title: 'Android Studio 설치 경로',
        trouble3Body: 'USB 디버깅을 켠 뒤 Android Studio에서 Run을 누르면 companion 앱을 바로 설치할 수 있습니다.',
        trouble4Title: '인텐트 액션 테스트',
        trouble4Body: '앱이 연결된 뒤 웹에서 액션을 전송하면, Android 쪽 서비스가 인텐트를 받아 실제 앱 실행 또는 액션 수행을 시도합니다.',
        techEyebrow: 'Technology Overview',
        techTitle: '기술 개요와 처리 파이프라인',
        techBody: '연구성 설명은 유지하되, 처음 보는 사람도 빠르게 이해할 수 있도록 핵심만 앞에 배치했습니다.',
        tech1Label: 'FUSION',
        tech1Title: '멀티 IMU 융합',
        tech1Body: '팔과 손목의 자세 변화를 함께 추적해 단일 센서 대비 더 안정적인 필기 경로를 확보합니다.',
        tech2Label: 'DRIFT',
        tech2Title: '드리프트 억제',
        tech2Body: '정지 구간과 보정 로직을 이용해 누적 오차를 줄이고, 문자가 무너지지 않도록 관리합니다.',
        tech3Label: 'PRODUCT',
        tech3Title: '공개 배포 친화적 UI',
        tech3Body: '실험 코드가 아니라 서비스형 진입 화면처럼 보이도록 데모/가이드/연결을 한 앱으로 묶었습니다.',
        pipelineTitle: 'End-to-end Pipeline',
        pipeline1: 'IMU Sampling',
        pipeline2: 'Sensor Fusion',
        pipeline3: 'Trajectory Projection',
        pipeline4: 'Recognition',
        pipeline5: 'Web + Android Action',
        detail1Title: '입력 구간 분리',
        detail1Body: '정지와 움직임 구간을 나눠 필기 시작/종료를 추정하고 의미 있는 스트로크만 남깁니다.',
        detail2Title: '평면 투영',
        detail2Body: '브라우저 데모에서는 획이 읽히게 보이는 것이 중요하므로, 시연용 평면 보드 투영을 사용합니다.',
        detail3Title: '후보 표시',
        detail3Body: '최상위 인식 결과만이 아니라 후보 리스트와 점수를 같이 보여 사용자가 결과를 해석할 수 있게 합니다.',
        detail4Title: '클라우드/로컬 분리',
        detail4Body: '공개 배포는 설명과 데모를 담당하고, 실제 센서 연결은 로컬 네트워크 환경에서 시도하도록 경계를 나눕니다.',
        teamEyebrow: 'Project Owner',
        teamTitle: '공개 가능한 실제 프로젝트 정보만 표시합니다',
        teamBody: '가짜 팀 카드 대신, 현재 공개 저장소 기준으로 확인 가능한 프로젝트 소유자와 저장소 링크만 유지했습니다.',
        teamRole: '프로젝트 오너 / 메인 개발자',
        teamBio: 'AirWriting 저장소를 유지하며 웹, Android companion, 실험 파이프라인을 함께 다루는 공개 프로젝트 소유자입니다.',
        teamGithub: 'GitHub 프로필',
        teamRepo: '저장소 보기',
        contactEyebrow: 'Contact / Feedback',
        contactTitle: '질문과 피드백을 남길 수 있는 공간',
        contactBody: '공개 배포에서도 댓글 API가 살아 있으면 피드백을 받을 수 있고, 실패하더라도 이유를 화면에 명시합니다.',
        contactFormLabel: 'Write Feedback',
        contactFormTitle: '간단한 문의 남기기',
        contactNameLabel: '이름',
        contactMessageLabel: '메시지',
        contactNamePlaceholder: '이름 또는 닉네임',
        contactMessagePlaceholder: '질문, 개선 요청, 사용 시나리오를 남겨주세요.',
        contactSubmit: '보내기',
        contactRecentLabel: 'Recent Messages',
        footerBody: '센서 기반 공중 필기를 제품형 경험으로 정리한 공개 플랫폼 데모',
        footerSource: '소스 코드',
        commentsLoading: '댓글을 불러오는 중입니다.',
        commentsEmpty: '아직 등록된 메시지가 없습니다. 첫 피드백을 남겨보세요.',
        commentsLoaded: '최근 피드백',
        commentsFailed: '댓글 API에 연결하지 못했습니다.',
        commentPosting: '메시지를 전송하는 중입니다.',
        commentPosted: '메시지를 남겼습니다.',
        commentPostFailed: '메시지를 전송하지 못했습니다.',
        commentValidation: '이름과 메시지를 모두 입력해주세요.',
        demoModeBadge: 'DEMO',
        liveModeBadge: 'LIVE',
        demoSummary: '데모 모드는 배포 환경 기본값입니다. 미리 준비된 문자 궤적과 파형을 재생해 화면이 비지 않도록 구성했습니다.',
        liveSummaryReady: '라이브 모드는 로컬 WebSocket relay가 준비된 경우에만 권장됩니다. 연결에 실패하면 언제든 데모로 돌아갈 수 있습니다.',
        liveSummaryConnecting: '로컬 relay에 연결을 시도하는 중입니다. 준비되지 않았다면 몇 초 뒤 자동으로 데모 복귀 안내가 뜹니다.',
        liveSummaryFailed: '라이브 relay를 찾지 못했습니다. 공개 배포에서는 정상이며, 로컬 실험 환경에서 다시 시도하세요.',
        bannerDemoTitle: '배포 기본 화면',
        bannerDemoText: '문자 형태가 보이도록 정리된 시연용 스트로크를 재생 중입니다.',
        bannerDemoAction: '라이브 모드',
        bannerLiveTitle: '로컬 장비 연결 대기',
        bannerLiveText: 'WebSocket 연결을 시도합니다. relay가 없으면 데모 모드로 돌아갈 수 있습니다.',
        bannerLiveAction: '데모로 복귀',
        bannerLiveFailedTitle: '라이브 연결 실패',
        bannerLiveFailedText: '클라우드 배포에서는 정상입니다. 로컬 PC와 Android 앱을 같은 네트워크에서 다시 연결하세요.',
        bannerLiveFailedAction: '데모 재개',
        pairingPublicMode: 'PUBLIC HOST',
        pairingLocalMode: 'LOCAL HOST',
        pairingPublicState: '공개 배포 주소',
        pairingLocalState: '로컬 연결 가능',
        pairingPublicHelp: '현재 페이지는 공개 서버에서 열려 있습니다. 휴대폰 앱은 Render 주소가 아니라 로컬 PC에서 실행 중인 relay 주소에 연결되어야 합니다.',
        pairingLocalHelp: '같은 Wi-Fi의 Android 앱에서 아래 주소를 스캔하거나 직접 입력하세요.',
        pairingRefreshing: '연결 주소를 확인하는 중입니다.',
        pairingUnavailable: '호스트 주소를 확인하지 못했습니다.',
        liveConnecting: 'CONNECTING',
        liveConnected: 'CONNECTED',
        liveDisconnected: 'DISCONNECTED',
        zuptWriting: 'WRITING',
        zuptStable: 'STABLE',
        demoRecognized: '인식됨'
    },
    en: {
        introKicker: 'AIRWRITING PLATFORM',
        introTitle: 'Turning motion in the air into readable writing.',
        introSubtitle: 'A public AirWriting demo that brings three IMUs, real-time estimation, and browser visualization into one deployable surface.',
        introSkip: 'Skip intro',
        brandTitle: 'Deployable AirWriting Platform',
        navHome: 'Home',
        navStudio: 'Studio',
        navAndroid: 'Connect Android',
        navTechnology: 'Technology',
        navTeam: 'Team',
        navContact: 'Contact',
        heroEyebrow: 'Public Demo / Product Surface',
        heroTitle: 'Not just motion data, but a writing experience that actually looks readable.',
        heroBody: 'AirWriting stabilizes IMU-based writing in the air and presents demo, live, and installation guidance in one browser surface. It should make sense on a public deployment and still support real device pairing in local mode.',
        heroCtaDemo: 'Open Demo Studio',
        heroCtaLive: 'Open Live Studio',
        heroCtaAndroid: 'Install / Pair Android',
        heroDemoLabel: 'DEPLOY MODE',
        heroDemoTitle: 'Instant demo without hardware',
        heroDemoBody: 'On the public deployment, prebuilt stroke samples replay by default so visitors immediately see legible writing.',
        heroLiveLabel: 'LOCAL MODE',
        heroLiveTitle: 'Switch to live when hardware is ready',
        heroLiveBody: 'When your PC and Android phone share the same network, you can attempt a WebSocket-backed local live session.',
        heroSurfaceLabel: 'Readable Demo Strokes',
        heroSurfaceBody: 'The demo projects strokes onto a flat board so letters stay crisp instead of dissolving into scribbles.',
        metricAccuracy: 'best recognition accuracy',
        metricRealtime: 'real-time sample rate',
        metricSensors: 'active sensors',
        proofEyebrow: 'Why This Feels Like A Product',
        proofTitle: 'Structured so the public deployment shows value immediately',
        proof1Label: '01',
        proof1Title: 'Legible demo writing',
        proof1Body: 'The default playback uses deliberate letter paths rather than noisy wandering strokes.',
        proof2Label: '02',
        proof2Title: 'Live connection is separated',
        proof2Body: 'Public deployment and local experimentation are split so the screen never feels empty when hardware is absent.',
        proof3Label: '03',
        proof3Title: 'Install-to-pair journey',
        proof3Body: 'Android installation, developer mode, QR/IP pairing, and troubleshooting are collected into one guided surface.',
        flowEyebrow: 'How It Works',
        flowTitle: 'Visitors only need to understand three steps',
        flow1Title: 'Capture sensors and writing segments',
        flow1Body: 'Three IMUs sample arm and hand motion while the system isolates the writing window.',
        flow2Title: 'Estimate pose and stabilize drift',
        flow2Body: 'Drift suppression and pose estimation re-align the motion into a cleaner writing plane.',
        flow3Title: 'Visualize and trigger actions',
        flow3Body: 'The browser displays recognition and telemetry, while the Android app serves as the action endpoint.',
        pathsEyebrow: 'Choose Your Path',
        pathsTitle: 'Separate entry points based on user intent',
        pathsDemoTitle: 'Public demo',
        pathsDemoBody: 'See a working writing demo and product UI even without hardware.',
        pathsDemoButton: 'See demo',
        pathsLiveTitle: 'Local live mode',
        pathsLiveBody: 'Run your relay/action dispatcher locally, then try the real sensor stream.',
        pathsLiveButton: 'Try live mode',
        studioEyebrow: 'Studio Surface',
        studioTitle: 'Real-time / demo air-writing studio',
        modeDemo: 'Demo Mode',
        modeLive: 'Live Mode',
        recognitionLabel: 'Recognition',
        sessionLabel: 'Session Control',
        sessionSafe: 'Demo is default on deploy',
        sessionRunDemo: 'Replay demo',
        sessionTryLive: 'Try live connection',
        sessionBackDemo: 'Back to demo',
        guideLabel: 'Quick Guide',
        guideWhenLive: 'Before live mode',
        guideStep1: 'Put the PC and Android device on the same Wi-Fi.',
        guideStep2: 'Make sure the relay/action dispatcher is running on your PC.',
        guideStep3: 'If the connection fails, return to demo mode and validate the UI first.',
        boardCaption: 'Readable demo strokes projected onto a flat board',
        writingState: 'WRITING',
        statusConnection: 'Connection',
        statusPointer: 'Pointer',
        statusZupt: 'State',
        statusWord: 'Current word',
        waveLabel: 'Waveform',
        waveTitle: 'Pitch / Roll / Yaw',
        deployLabel: 'Public Deployment',
        deployTitle: 'Deployment behavior',
        deployBody: 'On public Render, demo mode is the default. Only use live mode when your local hardware and relay are actually ready.',
        androidEyebrow: 'Android Companion',
        androidTitle: 'Android installation and pairing guide',
        androidBody: 'The web app handles visualization and configuration, while the Android app acts as the execution endpoint. Public deploy explains the structure; local mode enables real QR/IP pairing.',
        androidCard1Label: 'APP ROLE',
        androidCard1Title: 'Action execution endpoint',
        androidCard1Body: 'The companion app receives actions from the web and maps them to Android intents or app launches.',
        androidCard2Label: 'PAIRING',
        androidCard2Title: 'QR or manual IP pairing',
        androidCard2Body: 'On the same Wi-Fi, scan the PC address or type it directly to connect over WebSocket.',
        androidCard3Label: 'SERVICE',
        androidCard3Title: 'Onboarding for actual users',
        androidCard3Body: 'Install flow, developer mode, debugging, and network checks are collected like a real product surface.',
        pairLabel: 'Pairing',
        pairUrlLabel: 'Target address',
        pairStateLabel: 'Host state',
        pairRefresh: 'Refresh address',
        pairOpenStudio: 'Open studio',
        installGuideLabel: 'Install Guide',
        installGuideTitle: 'Practical setup order',
        install1Title: 'Open or install the Android app',
        install1Body: 'Run the `android_app` module from Android Studio or prepare a future APK build.',
        install2Title: 'Enable developer options',
        install2Body: 'Tap Build Number multiple times in Android settings to unlock developer options.',
        install3Title: 'Allow USB debugging',
        install3Body: 'If you install from Android Studio, enable USB debugging and trust the connected PC.',
        install4Title: 'Allow installs from unknown sources',
        install4Body: 'If you install from an APK file, grant install permission to your browser or file manager.',
        install5Title: 'Join the same Wi-Fi',
        install5Body: 'The phone and PC must share the same local network for `ws://<PC_IP>:18800` to work.',
        install6Title: 'Scan QR or enter IP',
        install6Body: 'In the app pairing screen, scan the QR code or type the shown IP directly.',
        trouble1Title: 'QR exists but the phone will not connect',
        trouble1Body: 'Check that both devices are on the same router and that the PC firewall is not blocking port 18800.',
        trouble2Title: 'Public deploy address is not your PC address',
        trouble2Body: 'The Render URL points to the cloud host. Your phone must connect to the relay running on your local PC.',
        trouble3Title: 'Installing through Android Studio',
        trouble3Body: 'With USB debugging enabled, pressing Run in Android Studio installs the companion app directly.',
        trouble4Title: 'Testing intent actions',
        trouble4Body: 'Once paired, web-triggered actions should be received by the Android service and translated into app launches or intents.',
        techEyebrow: 'Technology Overview',
        techTitle: 'Pipeline and technical framing',
        techBody: 'The research explanation stays, but the first layer is condensed for people seeing the project for the first time.',
        tech1Label: 'FUSION',
        tech1Title: 'Multi-IMU fusion',
        tech1Body: 'Tracking both arm and wrist motion improves stability over a single-sensor writing path.',
        tech2Label: 'DRIFT',
        tech2Title: 'Drift suppression',
        tech2Body: 'Static windows and compensation logic help prevent the path from collapsing over time.',
        tech3Label: 'PRODUCT',
        tech3Title: 'Public-deploy friendly UI',
        tech3Body: 'The experience is framed like a product surface instead of a raw experiment dashboard.',
        pipelineTitle: 'End-to-end Pipeline',
        pipeline1: 'IMU Sampling',
        pipeline2: 'Sensor Fusion',
        pipeline3: 'Trajectory Projection',
        pipeline4: 'Recognition',
        pipeline5: 'Web + Android Action',
        detail1Title: 'Segment isolation',
        detail1Body: 'Start and end of writing are estimated so only meaningful strokes are retained.',
        detail2Title: 'Planar projection',
        detail2Body: 'The browser demo prioritizes readability, so the public playback uses a flat presentation board.',
        detail3Title: 'Candidate display',
        detail3Body: 'Not just the top recognition result but a ranked list and score are shown to help interpret the system.',
        detail4Title: 'Cloud/local separation',
        detail4Body: 'The public deployment focuses on explanation and demo, while real sensor connectivity remains a local-network feature.',
        teamEyebrow: 'Project Owner',
        teamTitle: 'Only real public project information is shown',
        teamBody: 'Instead of placeholders, the page now keeps only the owner and repository links visible in the public GitHub project.',
        teamRole: 'Project owner / primary developer',
        teamBio: 'Maintains the AirWriting repository across the web surface, Android companion, and experimental pipeline.',
        teamGithub: 'GitHub profile',
        teamRepo: 'View repository',
        contactEyebrow: 'Contact / Feedback',
        contactTitle: 'Space for questions and feedback',
        contactBody: 'If the comment API is available, visitors can leave feedback. If not, the UI explains the failure clearly.',
        contactFormLabel: 'Write Feedback',
        contactFormTitle: 'Leave a short message',
        contactNameLabel: 'Name',
        contactMessageLabel: 'Message',
        contactNamePlaceholder: 'Name or nickname',
        contactMessagePlaceholder: 'Leave a question, improvement request, or usage idea.',
        contactSubmit: 'Send',
        contactRecentLabel: 'Recent Messages',
        footerBody: 'A public platform demo that turns sensor-based air writing into a product-like experience',
        footerSource: 'Source code',
        commentsLoading: 'Loading messages.',
        commentsEmpty: 'No messages yet. Leave the first one.',
        commentsLoaded: 'Recent feedback',
        commentsFailed: 'Unable to reach the comments API.',
        commentPosting: 'Sending your message.',
        commentPosted: 'Message sent.',
        commentPostFailed: 'Unable to send your message.',
        commentValidation: 'Please enter both a name and a message.',
        demoModeBadge: 'DEMO',
        liveModeBadge: 'LIVE',
        demoSummary: 'Demo mode is the deployment default. Prepared stroke sequences and waveform data keep the page useful even without hardware.',
        liveSummaryReady: 'Live mode is only recommended when your local WebSocket relay is ready. You can always fall back to demo mode.',
        liveSummaryConnecting: 'Trying to connect to the local relay. If it is not running, a fallback prompt will appear shortly.',
        liveSummaryFailed: 'Live relay was not found. This is normal on public deploy; retry from your local experiment environment.',
        bannerDemoTitle: 'Public default surface',
        bannerDemoText: 'Playing curated writing strokes that remain readable as actual letters.',
        bannerDemoAction: 'Live mode',
        bannerLiveTitle: 'Waiting for local device connection',
        bannerLiveText: 'Attempting WebSocket connection. If the relay is unavailable, return to demo mode.',
        bannerLiveAction: 'Back to demo',
        bannerLiveFailedTitle: 'Live connection failed',
        bannerLiveFailedText: 'This is expected on a cloud deployment. Reconnect from a local PC and Android device on the same network.',
        bannerLiveFailedAction: 'Resume demo',
        pairingPublicMode: 'PUBLIC HOST',
        pairingLocalMode: 'LOCAL HOST',
        pairingPublicState: 'public deployment address',
        pairingLocalState: 'local pairing available',
        pairingPublicHelp: 'This page is open on a public server. Your phone should connect to the relay running on your local PC, not directly to Render.',
        pairingLocalHelp: 'On the same Wi-Fi, scan the address below from the Android app or type it manually.',
        pairingRefreshing: 'Checking host address.',
        pairingUnavailable: 'Unable to determine the host address.',
        liveConnecting: 'CONNECTING',
        liveConnected: 'CONNECTED',
        liveDisconnected: 'DISCONNECTED',
        zuptWriting: 'WRITING',
        zuptStable: 'STABLE',
        demoRecognized: 'Recognized'
    }
};

const demoWords = [
    { label: 'AIR', score: 97.2, candidates: [{ label: 'AIR', score: 97.2 }, { label: 'AIM', score: 92.6 }, { label: 'AR', score: 86.9 }], path: [...letterA(-210, 0, 1.2), ...letterI(-30, 0, 1.2), ...letterR(110, 0, 1.2)] },
    { label: 'IMU', score: 98.1, candidates: [{ label: 'IMU', score: 98.1 }, { label: 'LMU', score: 91.4 }, { label: 'INU', score: 87.7 }], path: [...letterI(-210, 0, 1.18), ...letterM(-30, 0, 1.18), ...letterU(150, 0, 1.18)] },
    { label: 'DRIFT', score: 94.8, candidates: [{ label: 'DRIFT', score: 94.8 }, { label: 'DRIFTS', score: 88.3 }, { label: 'SHIFT', score: 82.1 }], path: [...letterD(-295, 0, 0.95), ...letterR(-155, 0, 0.95), ...letterI(-25, 0, 0.95), ...letterF(65, 0, 0.95), ...letterT(175, 0, 0.95)] }
];

const studioState = { language: 'ko', mode: 'demo', ws: null, liveAttemptTimer: null, demoFrame: 0, demoWordIndex: 0, animationHandle: null, currentPath: demoWords[0].path, projectedPoints: [], currentWord: demoWords[0], isWriting: false, waveHistory: { pitch: [], roll: [], yaw: [] } };

const elements = {
    tabs: Array.from(document.querySelectorAll('.nav-link')),
    pages: Array.from(document.querySelectorAll('.page')),
    langToggle: document.getElementById('langToggle'),
    introOverlay: document.getElementById('introOverlay'),
    introSkip: document.getElementById('introSkip'),
    modeDemoBtn: document.getElementById('modeDemoBtn'),
    modeLiveBtn: document.getElementById('modeLiveBtn'),
    btnRunDemo: document.getElementById('btnRunDemo'),
    btnTryLive: document.getElementById('btnTryLive'),
    btnFallbackDemo: document.getElementById('btnFallbackDemo'),
    drawingCanvas: document.getElementById('drawingCanvas'),
    waveformCanvas: document.getElementById('waveformCanvas'),
    stageBannerTitle: document.getElementById('stageBannerTitle'),
    stageBannerText: document.getElementById('stageBannerText'),
    stageBannerAction: document.getElementById('stageBannerAction'),
    recordingOverlay: document.getElementById('recordingOverlay'),
    recognizedTextOverlay: document.getElementById('recognizedTextOverlay'),
    aiResultWord: document.getElementById('aiResultWord'),
    aiResultScore: document.getElementById('aiResultScore'),
    aiCandidates: document.getElementById('aiCandidates'),
    studioModeBadge: document.getElementById('studioModeBadge'),
    modeSummary: document.getElementById('modeSummary'),
    valConn: document.getElementById('valConn'),
    valPos: document.getElementById('valPos'),
    valZupt: document.getElementById('valZupt'),
    studioWordLabel: document.getElementById('studioWordLabel'),
    commentsList: document.getElementById('commentsList'),
    commentsStatus: document.getElementById('commentsStatus'),
    commentForm: document.getElementById('commentForm'),
    commentFormStatus: document.getElementById('commentFormStatus'),
    submitCommentBtn: document.getElementById('submitCommentBtn'),
    commentAuthor: document.getElementById('commentAuthor'),
    commentContent: document.getElementById('commentContent'),
    pairingModeLabel: document.getElementById('pairingModeLabel'),
    pairingUrl: document.getElementById('pairingUrl'),
    pairingHostState: document.getElementById('pairingHostState'),
    pairingQr: document.getElementById('pairingQr'),
    pairingHelpText: document.getElementById('pairingHelpText'),
    refreshPairingBtn: document.getElementById('refreshPairingBtn')
};

const drawCtx = elements.drawingCanvas.getContext('2d');
const waveformCtx = elements.waveformCanvas.getContext('2d');

document.addEventListener('DOMContentLoaded', () => {
    applyTranslations();
    bindEvents();
    resizeCanvases();
    initializeIntro();
    selectTab('tab-home');
    switchMode('demo');
    loadComments();
    refreshPairingInfo();
    window.addEventListener('resize', resizeCanvases);
});

function bindEvents() {
    elements.tabs.forEach((tab) => tab.addEventListener('click', () => selectTab(tab.dataset.target)));
    document.querySelectorAll('[data-target-jump]').forEach((button) => button.addEventListener('click', () => selectTab(button.dataset.targetJump)));
    document.querySelectorAll('[data-open-studio]').forEach((button) => button.addEventListener('click', () => { selectTab('tab-studio'); switchMode(button.dataset.openStudio === 'live' ? 'live' : 'demo'); }));
    elements.langToggle.addEventListener('click', () => {
        studioState.language = studioState.language === 'ko' ? 'en' : 'ko';
        applyTranslations();
        renderRecognition(studioState.currentWord);
        updateStageBanner();
        updateModeSummary();
        updateConnectionStatus(studioState.mode === 'live' && studioState.ws && studioState.ws.readyState === WebSocket.OPEN ? t('liveConnected') : studioState.mode === 'live' ? t('liveConnecting') : t('liveDisconnected'));
        refreshPairingInfo();
        loadComments();
    });
    elements.introSkip.addEventListener('click', closeIntro);
    elements.modeDemoBtn.addEventListener('click', () => switchMode('demo'));
    elements.modeLiveBtn.addEventListener('click', () => switchMode('live'));
    elements.btnRunDemo.addEventListener('click', () => { switchMode('demo'); restartDemo(true); });
    elements.btnTryLive.addEventListener('click', () => switchMode('live'));
    elements.btnFallbackDemo.addEventListener('click', () => switchMode('demo'));
    elements.stageBannerAction.addEventListener('click', () => switchMode(studioState.mode === 'demo' ? 'live' : 'demo'));
    elements.refreshPairingBtn.addEventListener('click', refreshPairingInfo);
    elements.commentForm.addEventListener('submit', handleCommentSubmit);
}

function t(key) { return translations[studioState.language][key] ?? translations.ko[key] ?? key; }
function applyTranslations() {
    document.documentElement.lang = studioState.language;
    document.querySelectorAll('[data-i18n]').forEach((node) => { node.textContent = t(node.dataset.i18n); });
    document.querySelectorAll('[data-i18n-placeholder]').forEach((node) => node.setAttribute('placeholder', t(node.dataset.i18nPlaceholder)));
    elements.langToggle.textContent = studioState.language === 'ko' ? 'EN' : 'KO';
    document.title = studioState.language === 'ko' ? 'AirWriting | 공중 필기 플랫폼' : 'AirWriting | Air-writing platform';
}
function initializeIntro() { if (sessionStorage.getItem('airwriting_intro_seen') === '1') { elements.introOverlay.classList.add('hidden'); return; } window.setTimeout(closeIntro, 2600); }
function closeIntro() { elements.introOverlay.classList.add('hidden'); sessionStorage.setItem('airwriting_intro_seen', '1'); }
function selectTab(tabId) { elements.tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.target === tabId)); elements.pages.forEach((page) => page.classList.toggle('active', page.id === tabId)); if (tabId === 'tab-studio') { resizeCanvases(); renderStudioFrame(); } }
function switchMode(mode) {
    clearLiveAttempt(); closeSocket(); studioState.mode = mode;
    elements.modeDemoBtn.classList.toggle('active', mode === 'demo'); elements.modeLiveBtn.classList.toggle('active', mode === 'live');
    if (mode === 'demo') { elements.studioModeBadge.textContent = t('demoModeBadge'); updateConnectionStatus(t('liveDisconnected')); updateModeSummary(); updateStageBanner(); restartDemo(true); return; }
    elements.studioModeBadge.textContent = t('liveModeBadge'); updateConnectionStatus(t('liveConnecting')); updateModeSummary('connecting'); updateStageBanner('connecting'); beginLiveAttempt();
}
function updateModeSummary(state = '') { elements.modeSummary.textContent = studioState.mode === 'demo' ? t('demoSummary') : state === 'failed' ? t('liveSummaryFailed') : state === 'connecting' ? t('liveSummaryConnecting') : t('liveSummaryReady'); }
function updateStageBanner(state = '') {
    if (studioState.mode === 'demo') { elements.stageBannerTitle.textContent = t('bannerDemoTitle'); elements.stageBannerText.textContent = t('bannerDemoText'); elements.stageBannerAction.textContent = t('bannerDemoAction'); return; }
    if (state === 'failed') { elements.stageBannerTitle.textContent = t('bannerLiveFailedTitle'); elements.stageBannerText.textContent = t('bannerLiveFailedText'); elements.stageBannerAction.textContent = t('bannerLiveFailedAction'); return; }
    elements.stageBannerTitle.textContent = t('bannerLiveTitle'); elements.stageBannerText.textContent = t('bannerLiveText'); elements.stageBannerAction.textContent = t('bannerLiveAction');
}
function restartDemo(advanceWord) {
    if (advanceWord) studioState.demoWordIndex = (studioState.demoWordIndex + 1) % demoWords.length;
    studioState.currentWord = demoWords[studioState.demoWordIndex]; studioState.currentPath = studioState.currentWord.path; studioState.demoFrame = 0; studioState.projectedPoints = []; renderRecognition(studioState.currentWord); startAnimationLoop();
}
function startAnimationLoop() {
    if (studioState.animationHandle) cancelAnimationFrame(studioState.animationHandle);
    const loop = () => { renderStudioFrame(); studioState.animationHandle = requestAnimationFrame(loop); };
    loop();
}
function renderStudioFrame() { if (studioState.mode === 'demo') advanceDemo(); else if (!studioState.ws || studioState.ws.readyState !== WebSocket.OPEN) pulseIdleFrame(); drawBoard(); drawWaveform(); }
function advanceDemo() {
    const totalFrames = studioState.currentPath.length + 110; studioState.demoFrame = (studioState.demoFrame + 1) % totalFrames;
    const visibleCount = Math.min(studioState.demoFrame, studioState.currentPath.length); studioState.projectedPoints = studioState.currentPath.slice(0, visibleCount); studioState.isWriting = visibleCount > 0 && visibleCount < studioState.currentPath.length;
    if (visibleCount === studioState.currentPath.length && studioState.demoFrame === studioState.currentPath.length + 24) showRecognizedToast(studioState.currentWord.label);
    if (studioState.demoFrame === totalFrames - 1) { studioState.demoWordIndex = (studioState.demoWordIndex + 1) % demoWords.length; studioState.currentWord = demoWords[studioState.demoWordIndex]; studioState.currentPath = studioState.currentWord.path; studioState.projectedPoints = []; renderRecognition(studioState.currentWord); }
    driveTelemetry(visibleCount / Math.max(studioState.currentPath.length, 1));
}
function pulseIdleFrame() { const time = Date.now() * 0.0012; studioState.projectedPoints = Array.from({ length: 18 }, (_, index) => ({ x: -180 + index * 22, y: Math.sin(time + index * 0.24) * 8 })); studioState.isWriting = false; driveTelemetry(0.18 + Math.abs(Math.sin(time)) * 0.08); }
function driveTelemetry(progress) {
    const point = studioState.projectedPoints[studioState.projectedPoints.length - 1] || { x: 0, y: 0 };
    elements.valPos.textContent = `${Math.round(point.x)}, ${Math.round(point.y)}`; elements.valZupt.textContent = studioState.isWriting ? t('zuptWriting') : t('zuptStable'); elements.studioWordLabel.textContent = studioState.currentWord.label; elements.recordingOverlay.style.opacity = studioState.isWriting ? '1' : '0.35'; elements.recordingOverlay.textContent = t('writingState');
    const phase = progress * Math.PI * 2; pushWave('pitch', 18 + Math.sin(phase * 1.2) * 12 + Math.sin(phase * 3.2) * 4); pushWave('roll', Math.cos(phase * 1.4) * 16 + Math.sin(phase * 2.1) * 5); pushWave('yaw', Math.sin(phase * 0.8 + 0.7) * 20);
}
function pushWave(key, value) { const series = studioState.waveHistory[key]; series.push(value); if (series.length > 120) series.shift(); }
function renderRecognition(word) { elements.aiResultWord.textContent = word.label; elements.aiResultScore.textContent = `${word.score.toFixed(1)}%`; elements.aiCandidates.innerHTML = ''; word.candidates.forEach((candidate, index) => { const item = document.createElement('li'); item.className = 'candidate-item'; item.innerHTML = `<span>${index + 1}. ${candidate.label}</span><strong>${candidate.score.toFixed(1)}%</strong>`; elements.aiCandidates.appendChild(item); }); }
function showRecognizedToast(word) { elements.recognizedTextOverlay.textContent = `${t('demoRecognized')}: ${word}`; elements.recognizedTextOverlay.classList.add('show'); window.setTimeout(() => elements.recognizedTextOverlay.classList.remove('show'), 1200); }
function drawBoard() {
    const width = elements.drawingCanvas.width; const height = elements.drawingCanvas.height; drawCtx.clearRect(0, 0, width, height);
    const gradient = drawCtx.createLinearGradient(0, 0, width, height); gradient.addColorStop(0, 'rgba(9, 18, 33, 0.94)'); gradient.addColorStop(1, 'rgba(20, 44, 76, 0.9)'); drawCtx.fillStyle = gradient; drawCtx.fillRect(0, 0, width, height); drawBoardGrid(width, height);
    drawCtx.save(); drawCtx.translate(width / 2, height / 2 + 18); drawCtx.strokeStyle = 'rgba(129, 210, 255, 0.18)'; drawCtx.lineWidth = 2; drawCtx.strokeRect(-width * 0.32, -height * 0.2, width * 0.64, height * 0.42);
    if (studioState.projectedPoints.length > 1) { drawStroke('rgba(120, 234, 255, 0.25)', 26, 0); drawStroke('rgba(197, 245, 255, 0.94)', 10, 18); const tip = studioState.projectedPoints[studioState.projectedPoints.length - 1]; drawCtx.shadowBlur = 28; drawCtx.fillStyle = '#ffffff'; drawCtx.beginPath(); drawCtx.arc(tip.x, tip.y, 8, 0, Math.PI * 2); drawCtx.fill(); }
    drawCtx.restore();
}
function drawStroke(color, width, blur) { drawCtx.beginPath(); studioState.projectedPoints.forEach((point, index) => { if (index === 0) drawCtx.moveTo(point.x, point.y); else drawCtx.lineTo(point.x, point.y); }); drawCtx.strokeStyle = color; drawCtx.lineWidth = width; drawCtx.lineCap = 'round'; drawCtx.lineJoin = 'round'; drawCtx.shadowColor = 'rgba(127, 224, 255, 0.45)'; drawCtx.shadowBlur = blur; drawCtx.stroke(); }
function drawBoardGrid(width, height) { drawCtx.save(); drawCtx.strokeStyle = 'rgba(142, 201, 255, 0.08)'; drawCtx.lineWidth = 1; for (let x = 0; x <= width; x += 48) { drawCtx.beginPath(); drawCtx.moveTo(x, 0); drawCtx.lineTo(x, height); drawCtx.stroke(); } for (let y = 0; y <= height; y += 48) { drawCtx.beginPath(); drawCtx.moveTo(0, y); drawCtx.lineTo(width, y); drawCtx.stroke(); } drawCtx.restore(); }
function drawWaveform() {
    const width = elements.waveformCanvas.width; const height = elements.waveformCanvas.height; waveformCtx.clearRect(0, 0, width, height);
    const gradient = waveformCtx.createLinearGradient(0, 0, 0, height); gradient.addColorStop(0, 'rgba(11, 26, 44, 0.95)'); gradient.addColorStop(1, 'rgba(4, 14, 24, 0.95)'); waveformCtx.fillStyle = gradient; waveformCtx.fillRect(0, 0, width, height);
    [40, 80, 120].forEach((y) => { waveformCtx.strokeStyle = 'rgba(128, 176, 219, 0.12)'; waveformCtx.beginPath(); waveformCtx.moveTo(0, y); waveformCtx.lineTo(width, y); waveformCtx.stroke(); });
    drawSeries(studioState.waveHistory.pitch, '#84d8ff'); drawSeries(studioState.waveHistory.roll, '#66ffc7'); drawSeries(studioState.waveHistory.yaw, '#ffc36f');
}
function drawSeries(series, color) { if (!series.length) return; waveformCtx.beginPath(); series.forEach((value, index) => { const x = (index / Math.max(series.length - 1, 1)) * elements.waveformCanvas.width; const y = elements.waveformCanvas.height / 2 - value; if (index === 0) waveformCtx.moveTo(x, y); else waveformCtx.lineTo(x, y); }); waveformCtx.strokeStyle = color; waveformCtx.lineWidth = 2; waveformCtx.stroke(); }
function resizeCanvases() { resizeCanvas(elements.drawingCanvas, drawCtx); resizeCanvas(elements.waveformCanvas, waveformCtx); renderStudioFrame(); }
function resizeCanvas(canvas, ctx) { const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; const width = Math.max(Math.floor(rect.width * dpr), 1); const height = Math.max(Math.floor(rect.height * dpr), 1); if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); } }
function beginLiveAttempt() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'; const defaultPort = isLocalHost(window.location.hostname) ? ':18765' : ''; const wsUrl = `${protocol}//${window.location.hostname}${defaultPort}/ws`;
    try { studioState.ws = new WebSocket(wsUrl); } catch (error) { handleLiveFailure(); return; }
    studioState.ws.addEventListener('open', () => { updateConnectionStatus(t('liveConnected')); updateModeSummary(); updateStageBanner(); });
    studioState.ws.addEventListener('message', (event) => { try { ingestLivePayload(JSON.parse(event.data)); } catch (error) {} });
    studioState.ws.addEventListener('close', () => { if (studioState.mode === 'live') handleLiveFailure(); });
    studioState.ws.addEventListener('error', () => { if (studioState.mode === 'live') handleLiveFailure(); });
    studioState.liveAttemptTimer = window.setTimeout(() => { if (!studioState.ws || studioState.ws.readyState !== WebSocket.OPEN) handleLiveFailure(); }, 2600);
}
function ingestLivePayload(payload) {
    if (Array.isArray(payload.points) && payload.points.length) studioState.projectedPoints = payload.points.slice(-200).map((point) => ({ x: clamp(Number(point.x ?? point[0] ?? 0), -260, 260), y: clamp(Number(point.y ?? point[1] ?? 0), -140, 140) }));
    if (typeof payload.word === 'string' && payload.word.trim()) { studioState.currentWord = { label: payload.word.trim(), score: Number(payload.score ?? 94.3), candidates: Array.isArray(payload.candidates) && payload.candidates.length ? payload.candidates.map((candidate) => ({ label: String(candidate.label ?? candidate.word ?? candidate[0] ?? '').trim() || payload.word.trim(), score: Number(candidate.score ?? candidate[1] ?? 0) })) : [{ label: payload.word.trim(), score: Number(payload.score ?? 94.3) }] }; renderRecognition(studioState.currentWord); }
    studioState.isWriting = Boolean(payload.isWriting ?? payload.writing ?? (studioState.projectedPoints.length > 3));
    if (payload.telemetry) { pushWave('pitch', Number(payload.telemetry.pitch ?? 0)); pushWave('roll', Number(payload.telemetry.roll ?? 0)); pushWave('yaw', Number(payload.telemetry.yaw ?? 0)); } else { driveTelemetry(0.45); }
    const tip = studioState.projectedPoints[studioState.projectedPoints.length - 1] || { x: 0, y: 0 }; elements.valPos.textContent = `${Math.round(tip.x)}, ${Math.round(tip.y)}`; elements.valZupt.textContent = studioState.isWriting ? t('zuptWriting') : t('zuptStable'); elements.studioWordLabel.textContent = studioState.currentWord.label;
}
function handleLiveFailure() { clearLiveAttempt(); closeSocket(); if (studioState.mode !== 'live') return; updateConnectionStatus(t('liveDisconnected')); updateModeSummary('failed'); updateStageBanner('failed'); pulseIdleFrame(); drawBoard(); drawWaveform(); }
function clearLiveAttempt() { if (studioState.liveAttemptTimer) { clearTimeout(studioState.liveAttemptTimer); studioState.liveAttemptTimer = null; } }
function closeSocket() { if (!studioState.ws) return; try { studioState.ws.close(); } catch (error) {} studioState.ws = null; }
function updateConnectionStatus(text) { elements.valConn.textContent = text; }
async function refreshPairingInfo() {
    elements.pairingHelpText.textContent = t('pairingRefreshing'); elements.pairingUrl.textContent = 'ws://loading...:18800'; elements.pairingQr.innerHTML = '';
    if (!isLocalHost(window.location.hostname)) { elements.pairingModeLabel.textContent = t('pairingPublicMode'); elements.pairingHostState.textContent = t('pairingPublicState'); elements.pairingUrl.textContent = 'ws://<your-local-pc>:18800'; elements.pairingHelpText.textContent = t('pairingPublicHelp'); return; }
    try {
        const response = await fetch('/api/config/ip'); const payload = await response.json(); const ip = payload.ip || window.location.hostname; const wsUrl = `ws://${ip}:18800`;
        elements.pairingModeLabel.textContent = t('pairingLocalMode'); elements.pairingHostState.textContent = t('pairingLocalState'); elements.pairingUrl.textContent = wsUrl; elements.pairingHelpText.textContent = t('pairingLocalHelp');
        if (window.QRCode) new QRCode(elements.pairingQr, { text: wsUrl, width: 148, height: 148, colorDark: '#d8f7ff', colorLight: '#0a1421', correctLevel: QRCode.CorrectLevel.M });
    } catch (error) { elements.pairingModeLabel.textContent = t('pairingLocalMode'); elements.pairingHostState.textContent = t('pairingUnavailable'); elements.pairingUrl.textContent = 'ws://<local-ip>:18800'; elements.pairingHelpText.textContent = t('pairingUnavailable'); }
}
async function loadComments() {
    elements.commentsStatus.textContent = t('commentsLoading'); elements.commentsList.innerHTML = '';
    try {
        const response = await fetch('/api/comments'); if (!response.ok) throw new Error('failed'); const payload = await response.json(); const comments = Array.isArray(payload.comments) ? payload.comments : payload;
        if (!comments || comments.length === 0) { elements.commentsStatus.textContent = t('commentsEmpty'); return; }
        elements.commentsStatus.textContent = t('commentsLoaded'); comments.slice().reverse().slice(0, 6).forEach((comment) => elements.commentsList.appendChild(renderComment(comment)));
    } catch (error) { elements.commentsStatus.textContent = t('commentsFailed'); }
}
function renderComment(comment) { const item = document.createElement('article'); item.className = 'comment-item'; const author = escapeHtml(String(comment.author || comment.name || 'Anonymous')); const content = escapeHtml(String(comment.content || comment.message || '')); const createdAt = comment.created_at || comment.createdAt || comment.timestamp || ''; item.innerHTML = `<div class="comment-meta"><strong>${author}</strong><span>${escapeHtml(formatTimestamp(createdAt))}</span></div><p>${content}</p>`; return item; }
async function handleCommentSubmit(event) {
    event.preventDefault(); const author = elements.commentAuthor.value.trim(); const content = elements.commentContent.value.trim(); if (!author || !content) { elements.commentFormStatus.textContent = t('commentValidation'); return; }
    elements.commentFormStatus.textContent = t('commentPosting'); elements.submitCommentBtn.disabled = true;
    try {
        const response = await fetch('/api/comments', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ author, content }) }); if (!response.ok) throw new Error('failed');
        elements.commentAuthor.value = ''; elements.commentContent.value = ''; elements.commentFormStatus.textContent = t('commentPosted'); await loadComments();
    } catch (error) { elements.commentFormStatus.textContent = t('commentPostFailed'); } finally { elements.submitCommentBtn.disabled = false; }
}
function formatTimestamp(value) { if (!value) return studioState.language === 'ko' ? '방금 전' : 'just now'; const date = new Date(value); if (Number.isNaN(date.getTime())) return String(value); return new Intl.DateTimeFormat(studioState.language === 'ko' ? 'ko-KR' : 'en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date); }
function isLocalHost(hostname) { return hostname === 'localhost' || hostname === '127.0.0.1' || hostname.startsWith('192.168.') || hostname.startsWith('10.') || hostname.startsWith('172.'); }
function escapeHtml(value) { return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;'); }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function lineSegments(points, density = 10) { const path = []; for (let i = 0; i < points.length - 1; i += 1) { const [x1, y1] = points[i]; const [x2, y2] = points[i + 1]; for (let step = 0; step < density; step += 1) { const t = step / density; path.push({ x: x1 + (x2 - x1) * t, y: y1 + (y2 - y1) * t }); } } path.push({ x: points[points.length - 1][0], y: points[points.length - 1][1] }); return path; }
function arcPoints(cx, cy, rx, ry, start, end, density = 28) { const path = []; for (let i = 0; i <= density; i += 1) { const t = i / density; const angle = start + (end - start) * t; path.push({ x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry }); } return path; }
function joinStrokes(...strokes) { const joined = []; strokes.forEach((stroke, index) => { if (!stroke.length) return; if (index > 0 && joined.length) { const last = joined[joined.length - 1]; const first = stroke[0]; joined.push(...lineSegments([[last.x, last.y], [first.x, first.y]], 8)); } joined.push(...stroke); }); return joined; }
function scalePoints(points, offsetX, offsetY, scale) { return points.map((point) => ({ x: point.x * scale + offsetX, y: point.y * scale + offsetY })); }
function letterA(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-40, 70], [0, -70], [40, 70]], 10), lineSegments([[-22, 10], [22, 10]], 10)), offsetX, offsetY, scale); }
function letterI(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-28, -70], [28, -70]], 9), lineSegments([[0, -70], [0, 70]], 14), lineSegments([[-28, 70], [28, 70]], 9)), offsetX, offsetY, scale); }
function letterR(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-40, 70], [-40, -70]], 13), arcPoints(-10, -35, 38, 34, Math.PI, -Math.PI / 2, 24), lineSegments([[-10, 0], [42, 70]], 11)), offsetX, offsetY, scale); }
function letterM(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-50, 70], [-50, -70], [0, 10], [50, -70], [50, 70]], 11)), offsetX, offsetY, scale); }
function letterU(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-42, -70], [-42, 18]], 10), arcPoints(0, 18, 42, 50, Math.PI, 0, 24), lineSegments([[42, 18], [42, -70]], 10)), offsetX, offsetY, scale); }
function letterD(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-44, 70], [-44, -70]], 12), arcPoints(-10, 0, 50, 70, -Math.PI / 2, Math.PI / 2, 32), lineSegments([[-10, 70], [-44, 70]], 10), lineSegments([[-10, -70], [-44, -70]], 10)), offsetX, offsetY, scale); }
function letterF(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-35, -70], [-35, 70]], 14), lineSegments([[-35, -70], [38, -70]], 10), lineSegments([[-35, 2], [22, 2]], 10)), offsetX, offsetY, scale); }
function letterT(offsetX, offsetY, scale) { return scalePoints(joinStrokes(lineSegments([[-45, -70], [45, -70]], 12), lineSegments([[0, -70], [0, 70]], 14)), offsetX, offsetY, scale); }
