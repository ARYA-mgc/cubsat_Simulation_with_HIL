/**
 * @file    pid_embedded.h
 * @brief   CubeSat ACS — Embedded Cascaded PID Controller
 * @author  Arya MGC
 * @version 2.0.0
 *
 * Implements a fixed-point-friendly cascaded PID for STM32F4 / ARM Cortex-M4
 * compatible with 50 Hz HIL update rate.
 *
 * Usage:
 *   PID_Handle_t outer, inner;
 *   PID_Init(&outer, 2.80f, 0.35f, 1.10f, -50.0f, 50.0f, -5000.0f, 5000.0f);
 *   PID_Init(&inner, 5.50f, 0.80f, 0.40f, -1.5f,  1.5f,  -200.0f,  200.0f);
 *
 *   // In 50 Hz ISR:
 *   float vel_cmd = PID_Update(&outer, alt_ref, alt_meas, DT);
 *   float thrust  = PID_Update(&inner, vel_cmd,  vel_meas, DT);
 */

#ifndef PID_EMBEDDED_H
#define PID_EMBEDDED_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Configuration ────────────────────────────────────────────────────── */
#define PID_DT_HZ           50.0f       /**< Update rate [Hz]              */
#define PID_DT              (1.0f / PID_DT_HZ)
#define PID_MAX_AXES        3           /**< Roll, Pitch, Yaw              */

/* ─── PID Handle ───────────────────────────────────────────────────────── */
typedef struct {
    /* Gains */
    float kp;
    float ki;
    float kd;

    /* State */
    float integrator;
    float prev_error;
    float prev_measurement;   /**< For derivative-on-measurement filter    */

    /* Limits */
    float out_min;
    float out_max;
    float int_min;
    float int_max;

    /* Options */
    bool  derivative_on_measurement;   /**< Avoids derivative kick         */
    float d_filter_coeff;              /**< Low-pass on derivative, 0–1    */
    float d_filtered;

    /* Diagnostics */
    uint32_t update_count;
    float    last_output;
} PID_Handle_t;

/* ─── Attitude PID (3-axis) ────────────────────────────────────────────── */
typedef struct {
    PID_Handle_t axis[PID_MAX_AXES];   /**< [0]=Roll, [1]=Pitch, [2]=Yaw  */
} AttitudePID_t;

/* ─── Function Prototypes ──────────────────────────────────────────────── */

/**
 * @brief  Initialize PID controller
 * @param  pid      Pointer to PID handle
 * @param  kp       Proportional gain
 * @param  ki       Integral gain
 * @param  kd       Derivative gain
 * @param  out_min  Output lower saturation limit
 * @param  out_max  Output upper saturation limit
 * @param  int_min  Integrator lower clamp (anti-windup)
 * @param  int_max  Integrator upper clamp (anti-windup)
 */
void PID_Init(PID_Handle_t *pid,
              float kp, float ki, float kd,
              float out_min, float out_max,
              float int_min, float int_max);

/**
 * @brief  Compute PID output
 * @param  pid       Pointer to PID handle
 * @param  setpoint  Desired value
 * @param  measured  Measured (feedback) value
 * @param  dt        Time step [s]
 * @return PID output (saturated)
 */
float PID_Update(PID_Handle_t *pid,
                 float setpoint, float measured, float dt);

/**
 * @brief  Reset PID state (integrator + derivative memory)
 */
void PID_Reset(PID_Handle_t *pid);

/**
 * @brief  Enable/disable derivative-on-measurement mode
 */
void PID_SetDerivativeMode(PID_Handle_t *pid, bool on_measurement);

/**
 * @brief  Update gains at runtime (safe mid-flight)
 */
void PID_SetGains(PID_Handle_t *pid, float kp, float ki, float kd);

/**
 * @brief  Initialize 3-axis attitude PID with default gains
 */
void AttitudePID_Init(AttitudePID_t *apid);

/**
 * @brief  Compute 3-axis attitude torques
 * @param  apid     Pointer to attitude PID struct
 * @param  ref      Reference attitude [rad] — {roll, pitch, yaw}
 * @param  meas     Measured attitude [rad]
 * @param  dt       Time step [s]
 * @param  torque   Output torques [N·m] (must be float[3])
 */
void AttitudePID_Update(AttitudePID_t *apid,
                        const float ref[3], const float meas[3],
                        float dt, float torque[3]);

/* ─── Inline Clamp Utility ─────────────────────────────────────────────── */
static inline float _pid_clamp(float val, float lo, float hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

/* ─── Implementation (header-only for single-file convenience) ─────────── */
#ifdef PID_IMPLEMENTATION

void PID_Init(PID_Handle_t *pid,
              float kp, float ki, float kd,
              float out_min, float out_max,
              float int_min, float int_max)
{
    pid->kp           = kp;
    pid->ki           = ki;
    pid->kd           = kd;
    pid->out_min      = out_min;
    pid->out_max      = out_max;
    pid->int_min      = int_min;
    pid->int_max      = int_max;
    pid->integrator   = 0.0f;
    pid->prev_error   = 0.0f;
    pid->prev_measurement = 0.0f;
    pid->d_filtered   = 0.0f;
    pid->d_filter_coeff = 0.1f;   /* 10% low-pass weight */
    pid->derivative_on_measurement = true;
    pid->update_count = 0;
    pid->last_output  = 0.0f;
}

float PID_Update(PID_Handle_t *pid, float setpoint, float measured, float dt)
{
    float error = setpoint - measured;

    /* Proportional */
    float p = pid->kp * error;

    /* Integral with anti-windup clamp */
    pid->integrator = _pid_clamp(pid->integrator + pid->ki * error * dt,
                                  pid->int_min, pid->int_max);
    float i = pid->integrator;

    /* Derivative */
    float d_raw;
    if (pid->derivative_on_measurement) {
        /* Derivative-on-measurement: avoids kick on setpoint change */
        d_raw = -(measured - pid->prev_measurement) / dt;
        pid->prev_measurement = measured;
    } else {
        d_raw = (error - pid->prev_error) / dt;
    }

    /* Low-pass filter on derivative */
    pid->d_filtered = pid->d_filter_coeff * d_raw
                    + (1.0f - pid->d_filter_coeff) * pid->d_filtered;
    float d = pid->kd * pid->d_filtered;

    pid->prev_error = error;

    /* Sum and saturate */
    float output = _pid_clamp(p + i + d, pid->out_min, pid->out_max);
    pid->last_output = output;
    pid->update_count++;
    return output;
}

void PID_Reset(PID_Handle_t *pid) {
    pid->integrator       = 0.0f;
    pid->prev_error       = 0.0f;
    pid->prev_measurement = 0.0f;
    pid->d_filtered       = 0.0f;
}

void PID_SetDerivativeMode(PID_Handle_t *pid, bool on_measurement) {
    pid->derivative_on_measurement = on_measurement;
    PID_Reset(pid);
}

void PID_SetGains(PID_Handle_t *pid, float kp, float ki, float kd) {
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
}

void AttitudePID_Init(AttitudePID_t *apid) {
    /* Roll */
    PID_Init(&apid->axis[0], 12.0f, 0.5f, 2.8f,
             -0.01f, 0.01f, -0.5f, 0.5f);
    /* Pitch */
    PID_Init(&apid->axis[1], 12.0f, 0.5f, 2.8f,
             -0.01f, 0.01f, -0.5f, 0.5f);
    /* Yaw */
    PID_Init(&apid->axis[2],  8.0f, 0.3f, 1.5f,
             -0.01f, 0.01f, -0.5f, 0.5f);
    for (int i = 0; i < 3; i++)
        PID_SetDerivativeMode(&apid->axis[i], true);
}

void AttitudePID_Update(AttitudePID_t *apid,
                        const float ref[3], const float meas[3],
                        float dt, float torque[3])
{
    for (int i = 0; i < 3; i++) {
        torque[i] = PID_Update(&apid->axis[i], ref[i], meas[i], dt);
    }
}

#endif /* PID_IMPLEMENTATION */

#ifdef __cplusplus
}
#endif
#endif /* PID_EMBEDDED_H */
