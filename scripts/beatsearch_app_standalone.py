from beatsearch.utils import get_default_beatsearch_rhythms_fpath
from beatsearch.app.control import BSController
from beatsearch.app.view import BSApp

if __name__ == "__main__":
    controller = BSController()
    controller.set_corpus(get_default_beatsearch_rhythms_fpath())
    app = BSApp(controller, transport_frame_cls=None)
    app.mainloop()
