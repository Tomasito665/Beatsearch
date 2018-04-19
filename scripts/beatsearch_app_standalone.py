import logging
from beatsearch.utils import set_logging_level_by_name
from beatsearch.app.control import BSController
from beatsearch.app.view import BSApp
from beatsearch.app.config import get_argv_parser


if __name__ == "__main__":
    args = get_argv_parser().parse_args()
    set_logging_level_by_name(args.log)

    logging.debug("Initializing controller...")
    controller = BSController()

    logging.debug("Initializing view")
    app = BSApp(controller, transport_frame_cls=None)

    logging.debug("Initializing Tk event loop")
    app.mainloop()
