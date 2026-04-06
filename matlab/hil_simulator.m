% =========================================================================
%  Hardware-in-the-Loop (HIL) Simulation Interface
%  CubeSat ACS — MATLAB HIL Engine
%
%  Author : Arya MGC
%  Version: 2.0.0
%
%  Description:
%    Emulates real-time HIL communication between MATLAB plant model
%    and embedded flight computer. Sends control commands and receives
%    sensor feedback at 50 Hz via serial/UDP bridge.
%
%  HIL Architecture:
%    [MATLAB Plant Model] <—UDP/Serial—> [STM32 / Raspberry Pi CM4]
%                                              |
%                                        [NRF24L01+ Radio]
%                                              |
%                                        [SDR Telemetry]
% =========================================================================

classdef HILSimulator < handle

    properties
        % Communication
        udp_port     = 5005;
        udp_ip       = '127.0.0.1';
        serial_port  = 'COM3';          % Change for Linux: '/dev/ttyUSB0'
        baud_rate    = 115200;

        % Timing
        update_rate  = 50;              % Hz
        dt           = 0.02;            % s
        sim_time     = 0;

        % State
        altitude     = 400e3;           % m
        velocity     = 0;              % m/s
        attitude     = [0;0;0];        % rad [roll;pitch;yaw]
        ang_velocity = [0;0;0];        % rad/s

        % Telemetry
        packet_count = 0;
        signal_exchanges = 0;

        % Logging
        log_data     = struct();
        use_udp      = true;           % false = serial mode
    end

    methods
        %% Constructor
        function obj = HILSimulator(varargin)
            p = inputParser;
            addParameter(p, 'use_udp', true);
            addParameter(p, 'port', 5005);
            parse(p, varargin{:});
            obj.use_udp  = p.Results.use_udp;
            obj.udp_port = p.Results.port;

            obj.log_data.t          = [];
            obj.log_data.alt        = [];
            obj.log_data.vel        = [];
            obj.log_data.attitude   = [];
            obj.log_data.packets    = [];

            fprintf('[HIL] Simulator initialized at %d Hz\n', obj.update_rate);
        end

        %% Build sensor packet (mimics NRF24L01+ frame format)
        function packet = build_sensor_packet(obj)
            % Packet format (32 bytes — NRF max payload):
            % [0]    : Header  0xAE
            % [1]    : Seq num (uint8, wraps at 255)
            % [2-5]  : Altitude  float32
            % [6-9]  : Velocity  float32
            % [10-13]: Roll      float32
            % [14-17]: Pitch     float32
            % [18-21]: Yaw       float32
            % [22-25]: Timestamp uint32 (ms)
            % [26-29]: RSSI      float32 (dBm, simulated)
            % [30]   : Checksum  uint8
            % [31]   : Footer    0xEF

            seq_num   = mod(obj.packet_count, 256);
            timestamp = uint32(obj.sim_time * 1000);
            rssi_sim  = single(-65 + 5*randn());

            raw = [typecast(single(obj.altitude),    'uint8'), ...
                   typecast(single(obj.velocity),    'uint8'), ...
                   typecast(single(obj.attitude(1)), 'uint8'), ...
                   typecast(single(obj.attitude(2)), 'uint8'), ...
                   typecast(single(obj.attitude(3)), 'uint8'), ...
                   typecast(timestamp,               'uint8'), ...
                   typecast(rssi_sim,                'uint8')];

            checksum = mod(sum(raw), 256);
            packet   = [0xAE, seq_num, raw, checksum, 0xEF];
            obj.packet_count = obj.packet_count + 1;
        end

        %% Parse command packet from flight computer
        function [thrust, torque] = parse_command_packet(~, packet)
            % Command packet (20 bytes):
            % [0]   : Header 0xBC
            % [1-4] : Thrust   float32 [N]
            % [5-8] : Torque_x float32 [N·m]
            % [9-12]: Torque_y float32 [N·m]
            % [13-16]:Torque_z float32 [N·m]
            % [17]  : Checksum
            % [18]  : Footer 0xEF

            if length(packet) < 19 || packet(1) ~= 0xBC || packet(19) ~= 0xEF
                thrust = 0;  torque = [0;0;0];
                warning('[HIL] Malformed command packet — using zero control');
                return;
            end
            thrust   = double(typecast(uint8(packet(2:5)),  'single'));
            torque   = double([typecast(uint8(packet(6:9)),  'single'); ...
                               typecast(uint8(packet(10:13)),'single'); ...
                               typecast(uint8(packet(14:17)),'single')]);
        end

        %% Simulate one HIL step (plant integration)
        function step(obj, thrust, torque)
            m       = 1.33;
            drag    = 0.0012 * obj.velocity^2 * sign(obj.velocity);
            I_diag  = [1.2e-3; 1.2e-3; 0.8e-3];

            % Translational dynamics
            a          = (thrust - drag) / m;
            obj.velocity  = obj.velocity + a * obj.dt;
            obj.altitude  = obj.altitude + obj.velocity * obj.dt;

            % Rotational dynamics
            alpha         = torque ./ I_diag;
            obj.ang_velocity = obj.ang_velocity + alpha * obj.dt;
            obj.attitude     = obj.attitude + obj.ang_velocity * obj.dt;

            obj.sim_time  = obj.sim_time + obj.dt;
            obj.signal_exchanges = obj.signal_exchanges + 1;

            % Log
            obj.log_data.t        = [obj.log_data.t,        obj.sim_time];
            obj.log_data.alt      = [obj.log_data.alt,      obj.altitude];
            obj.log_data.vel      = [obj.log_data.vel,      obj.velocity];
            obj.log_data.attitude = [obj.log_data.attitude, obj.attitude];
        end

        %% Full HIL run loop (standalone / no real hardware)
        function run_standalone(obj, duration)
            fprintf('[HIL] Running standalone for %.0f s...\n', duration);
            N = round(duration / obj.dt);
            pid = CascadedPID();

            for k = 1:N
                ref_alt = 400e3 + (k > N/3)*20e3 - (k > 2*N/3)*10e3;
                sensor_pkt   = obj.build_sensor_packet();

                % Simulate flight computer PID (normally runs on embedded MCU)
                [thrust, torque] = pid.compute(ref_alt, [0;0;0], ...
                                                obj.altitude, obj.attitude, obj.dt);
                cmd_pkt = obj.build_command_packet(thrust, torque);

                % Plant step
                obj.step(thrust, torque);

                if mod(k, 100) == 0
                    fprintf('  t=%.1fs  Alt=%.1fkm  Vel=%.2fm/s  Pkts=%d\n', ...
                        obj.sim_time, obj.altitude/1e3, obj.velocity, obj.packet_count);
                end
            end
            fprintf('[HIL] Complete. Signal exchanges: %d\n', obj.signal_exchanges);
        end

        %% Build command packet
        function pkt = build_command_packet(~, thrust, torque)
            raw      = [typecast(single(thrust),    'uint8'), ...
                        typecast(single(torque(1)), 'uint8'), ...
                        typecast(single(torque(2)), 'uint8'), ...
                        typecast(single(torque(3)), 'uint8')];
            checksum = mod(sum(raw), 256);
            pkt      = [0xBC, raw, checksum, 0xEF];
        end

        %% Export results
        function export_csv(obj, filename)
            T = array2table([ obj.log_data.t', ...
                              obj.log_data.alt', ...
                              obj.log_data.vel', ...
                              obj.log_data.attitude'], ...
                'VariableNames', {'Time_s','Altitude_m','Velocity_ms', ...
                                  'Roll_rad','Pitch_rad','Yaw_rad'});
            writetable(T, filename);
            fprintf('[HIL] Exported %d rows to %s\n', height(T), filename);
        end
    end
end


%% ─── Cascaded PID (embedded in HIL for standalone mode) ─────────────────
classdef CascadedPID < handle
    properties
        int_alt = 0; err_alt_prev = 0;
        int_vel = 0; err_vel_prev = 0;
        int_att = [0;0;0]; err_att_prev = [0;0;0];
    end
    methods
        function [thrust, torque] = compute(obj, ref_alt, ref_att, alt, att, dt)
            % Outer: altitude → velocity command
            err_alt = ref_alt - alt;
            obj.int_alt = max(min(obj.int_alt + err_alt*dt, 5e3), -5e3);
            d_alt   = (err_alt - obj.err_alt_prev) / dt;
            vel_cmd = 2.80*err_alt + 0.35*obj.int_alt + 1.10*d_alt;
            vel_cmd = max(min(vel_cmd, 50), -50);
            obj.err_alt_prev = err_alt;

            % Inner: velocity → thrust
            err_vel = vel_cmd - 0;   % velocity approximated
            obj.int_vel = max(min(obj.int_vel + err_vel*dt, 200), -200);
            d_vel   = (err_vel - obj.err_vel_prev) / dt;
            thrust  = 1.33*(5.50*err_vel + 0.80*obj.int_vel + 0.40*d_vel);
            thrust  = max(min(thrust, 1.5), -1.5);
            obj.err_vel_prev = err_vel;

            % Attitude
            err_att = ref_att - att;
            obj.int_att = max(min(obj.int_att + err_att*dt, 0.5), -0.5);
            d_att   = (err_att - obj.err_att_prev) / dt;
            Kp = [12;12;8]; Ki = [0.5;0.5;0.3]; Kd = [2.8;2.8;1.5];
            torque  = Kp.*err_att + Ki.*obj.int_att + Kd.*d_att;
            torque  = max(min(torque, 0.01), -0.01);
            obj.err_att_prev = err_att;
        end
    end
end
