# GNSS Transmission Chain Simulation

This repository contains a Python simulation of a digital communication chain inspired by a **Global Navigation Satellite System (GNSS)**.  
The project models the end-to-end transmission of navigation data, from encoding and modulation to reception and demodulation.

The objective is to analyze how noise and Doppler effects impact the received signal and the symbol error rate (SER).

## Authors

Gabriel Casarotto  
Hippolyte André  

IPSA – Institut Polytechnique des Sciences Avancées  
Academic year 2025–2026

## Project Overview

This project implements a simplified GNSS communication chain composed of three main blocks:

Transmitter → Channel → Receiver

The simulation includes:

- Encoding of navigation data (latitude, longitude, altitude, timestamp)
- Digital modulation using **BPSK and QPSK**
- Transmission over a noisy channel
- Doppler frequency shift modelling
- Signal demodulation and symbol detection
- Performance evaluation through **symbol error rate (SER)**.

The system is implemented in Python using a baseband simulation framework. :contentReference[oaicite:0]{index=0}

## Transmitted Data

The transmitted message contains simulated GNSS parameters:

- Latitude
- Longitude
- Altitude
- Unix timestamp

Each parameter is converted to integers and encoded as binary vectors before modulation. :contentReference[oaicite:1]{index=1}

## Modulation Schemes

Two digital modulation techniques are implemented:

### BPSK (Binary Phase Shift Keying)

Each bit is mapped to a real symbol:

0 → −1  
1 → +1

### QPSK (Quadrature Phase Shift Keying)

Bits are grouped in pairs and mapped onto a complex constellation using Gray coding.

## Channel Model

The transmission channel includes several impairments.

### Additive White Gaussian Noise (AWGN)

The received signal is modeled as:

y(t) = x(t) + n(t)

where the noise follows a Gaussian distribution. :contentReference[oaicite:2]{index=2}

The simulation evaluates performance for different signal-to-noise ratios (SNR).

### Doppler Effect

To reproduce a realistic GNSS scenario, a Doppler frequency shift is introduced due to the relative motion between satellite and receiver.

This results in a **phase rotation in the complex baseband signal**, which affects constellation diagrams and detection performance. :contentReference[oaicite:3]{index=3}

## Receiver Architecture

The receiver processes the signal through several stages:

1. Down-conversion to baseband
2. Low-pass filtering
3. Matched filtering
4. Symbol sampling
5. Bit decision

The recovered symbols are then used to reconstruct the transmitted GNSS parameters.

## Performance Analysis

The system performance is evaluated using the **Symbol Error Rate (SER)**.

Simulation results are compared with theoretical expressions for BPSK and QPSK over an AWGN channel.

The project also studies:

- constellation diagrams
- time-domain signal representation
- effect of noise on navigation data reconstruction
- impact of Doppler frequency shift

## Repository Structure
code.py # Python implementation of the GNSS transmission chain

rapport.pdf # Project report

README.md

## Tools and Libraries

The simulation uses the Python scientific ecosystem:

- NumPy
- SciPy
- Matplotlib

## Academic Context

This project was completed as part of a **telecommunications course at IPSA**, focusing on digital communication systems used in satellite navigation.
