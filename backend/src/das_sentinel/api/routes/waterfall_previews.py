from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request, status

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.schemas import (
    WaterfallOptionsResponse,
    WaterfallPreviewRequest,
    WaterfallPreviewResponse,
)
from das_sentinel.services.waterfall_previews import (
    WaterfallPreviewConflict,
    WaterfallPreviewRangeError,
)
from das_sentinel.waterfalls import detector_display_name

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/{event_id}/waterfall-options", response_model=WaterfallOptionsResponse)
def waterfall_options(
    event_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> WaterfallOptionsResponse:
    del admin
    try:
        with request.app.state.session_factory() as session:
            options = request.app.state.waterfall_preview_service.options(session, event_id)
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found") from error
    except WaterfallPreviewConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Waterfall preview is unavailable") from error
    return WaterfallOptionsResponse(
        automatic_distance_start_m=options.automatic_distance_start_m,
        automatic_distance_stop_m=options.automatic_distance_stop_m,
        available_distance_start_m=options.available_distance_start_m,
        available_distance_stop_m=options.available_distance_stop_m,
        primary_detector=options.filter.detector,
        primary_detector_display=detector_display_name(options.filter.detector),
        filter_low_hz=options.filter.low_hz,
        filter_high_hz=options.filter.high_hz,
    )


@router.post("/{event_id}/waterfall-preview", response_model=WaterfallPreviewResponse)
def waterfall_preview(
    event_id: str,
    payload: WaterfallPreviewRequest,
    request: Request,
    admin: CsrfAdmin,
) -> WaterfallPreviewResponse:
    del admin
    try:
        with request.app.state.session_factory() as session:
            preview = request.app.state.waterfall_preview_service.render(
                session,
                event_id,
                payload.distance_start_m,
                payload.distance_stop_m,
            )
    except LookupError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found") from error
    except WaterfallPreviewRangeError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    except WaterfallPreviewConflict as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Waterfall preview is unavailable") from error
    return WaterfallPreviewResponse(
        distance_start_m=preview.distance_start_m,
        distance_stop_m=preview.distance_stop_m,
        channel_count=preview.channel_count,
        primary_detector=preview.filter.detector,
        primary_detector_display=detector_display_name(preview.filter.detector),
        filter_low_hz=preview.filter.low_hz,
        filter_high_hz=preview.filter.high_hz,
        raw_png_base64=base64.b64encode(preview.raw_png).decode("ascii"),
        filtered_png_base64=base64.b64encode(preview.filtered_png).decode("ascii"),
        waveform_png_base64=base64.b64encode(preview.waveform_png).decode("ascii"),
    )
